import uuid
import json
import time
import os
import logging
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Callable, Optional
from abc import ABC, abstractmethod

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MessageBus")

@dataclass
class Message:
    """Standard message format for A2A communication."""
    type: str
    payload: Dict[str, Any]
    sender: str = "unknown"
    receiver: str = "all"
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)
    
    def to_json(self) -> str:
        return json.dumps(asdict(self))
    
    @classmethod
    def from_json(cls, json_str: str) -> 'Message':
        data = json.loads(json_str)
        return cls(**data)

class MessageBus(ABC):
    """Abstract base class for Message Bus."""
    
    @abstractmethod
    def subscribe(self, message_type: str, callback: Callable[[Message], None]):
        pass
        
    @abstractmethod
    def publish(self, message: Message):
        pass
        
    @abstractmethod
    def start_consuming(self):
        """Start the consumer loop (blocking)."""
        pass

class InMemoryMessageBus(MessageBus):
    """Simple in-memory message bus for local development."""
    
    def __init__(self):
        self.subscribers: Dict[str, List[Callable[[Message], None]]] = {}
        self.history: List[Message] = []
        logger.info("Initialized InMemoryMessageBus")
        
    def subscribe(self, message_type: str, callback: Callable[[Message], None]):
        if message_type not in self.subscribers:
            self.subscribers[message_type] = []
        self.subscribers[message_type].append(callback)
        logger.info(f"Subscribed to '{message_type}'")
        
    def publish(self, message: Message):
        self.history.append(message)
        logger.info(f"Publishing message: {message.type} from {message.sender}")
        
        if message.type in self.subscribers:
            for callback in self.subscribers[message.type]:
                try:
                    callback(message)
                except Exception as e:
                    logger.error(f"Error in subscriber callback: {e}")
                    
    def start_consuming(self):
        """In-memory bus is synchronous, so this just waits."""
        logger.info("InMemoryMessageBus started (synchronous mode)")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass

class KafkaMessageBus(MessageBus):
    """Kafka-based message bus for distributed deployment."""
    
    def __init__(self, bootstrap_servers: str, group_id: str):
        try:
            from confluent_kafka import Producer, Consumer, KafkaError
        except ImportError:
            raise ImportError("confluent-kafka is required for KafkaMessageBus. Install it with 'pip install confluent-kafka'")
            
        self.bootstrap_servers = bootstrap_servers
        self.group_id = group_id
        self.callbacks: Dict[str, List[Callable[[Message], None]]] = {}
        self.running = False
        
        # Producer config
        self.producer = Producer({
            'bootstrap.servers': self.bootstrap_servers,
            'client.id': f'jolt-agent-{uuid.uuid4()}'
        })
        
        # Consumer config (will subscribe to all relevant topics)
        self.consumer = Consumer({
            'bootstrap.servers': self.bootstrap_servers,
            'group.id': self.group_id,
            'auto.offset.reset': 'earliest'
        })
        
        logger.info(f"Initialized KafkaMessageBus (servers={bootstrap_servers}, group={group_id})")

    def subscribe(self, message_type: str, callback: Callable[[Message], None]):
        if message_type not in self.callbacks:
            self.callbacks[message_type] = []
            
        self.callbacks[message_type].append(callback)
        
        # Subscribe to all topics we have callbacks for
        topics = list(self.callbacks.keys())
        self.consumer.subscribe(topics)
        logger.info(f"Subscribed to Kafka topic '{message_type}' (total topics: {topics})")

    def publish(self, message: Message):
        def delivery_report(err, msg):
            if err is not None:
                logger.error(f"Message delivery failed: {err}")
            else:
                logger.debug(f"Message delivered to {msg.topic()} [{msg.partition()}]")

        # Use message type as topic
        self.producer.produce(
            message.type,
            message.to_json().encode('utf-8'),
            callback=delivery_report
        )
        self.producer.flush() # Ensure it's sent
        logger.info(f"Published message to Kafka topic '{message.type}'")

    def start_consuming(self):
        """Start the Kafka consumer loop."""
        self.running = True
        logger.info("Starting Kafka consumer loop...")
        
        try:
            while self.running:
                msg = self.consumer.poll(1.0)

                if msg is None:
                    continue
                if msg.error():
                    logger.error(f"Consumer error: {msg.error()}")
                    continue

                try:
                    # Parse message
                    payload = msg.value().decode('utf-8')
                    message = Message.from_json(payload)
                    topic = msg.topic()
                    
                    logger.info(f"Received message: {message.type} from {message.sender}")
                    
                    # Dispatch to callbacks
                    if topic in self.callbacks:
                        for callback in self.callbacks[topic]:
                            try:
                                callback(message)
                            except Exception as e:
                                logger.error(f"Error in callback: {e}")
                                
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
                    
        except KeyboardInterrupt:
            pass
        finally:
            self.consumer.close()
            logger.info("Kafka consumer closed")

def get_message_bus() -> MessageBus:
    """Factory to get the appropriate MessageBus based on env vars."""
    kafka_servers = os.getenv("KAFKA_BOOTSTRAP_SERVERS")
    
    if kafka_servers:
        group_id = os.getenv("KAFKA_GROUP_ID", "jolt-platform-group")
        return KafkaMessageBus(kafka_servers, group_id)
    else:
        return InMemoryMessageBus()
