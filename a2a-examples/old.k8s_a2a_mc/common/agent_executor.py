import logging

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import (
    DataPart,
    InvalidParamsError,
    SendStreamingMessageSuccessResponse,
    Task,
    TaskArtifactUpdateEvent,
    TaskState,
    TaskStatusUpdateEvent,
    TextPart,
    UnsupportedOperationError,
)
from a2a.utils import new_agent_text_message, new_task
from a2a.utils.errors import ServerError
from common.base_agent import BaseAgent


logger = logging.getLogger(__name__)


class GenericAgentExecutor(AgentExecutor):
    """AgentExecutor used by the tragel agents."""

    def __init__(self, agent: BaseAgent):
        self.agent = agent

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        logger.info(f'Executing agent {self.agent.agent_name}')
        
        # Log the request context details
        logger.info(f'Request context: {context}')
        
        error = self._validate_request(context)
        if error:
            logger.error(f'Request validation error: {error}')
            raise ServerError(error=InvalidParamsError())

        query = context.get_user_input()
        logger.info(f'User input: {query}')

        task = context.current_task
        logger.info(f'Current task: {task}')

        if not task:
            logger.info('No current task, creating a new one')
            task = new_task(context.message)
            await event_queue.enqueue_event(task)

        updater = TaskUpdater(event_queue, task.id, task.contextId)

        logger.info('Starting to stream from agent')
        try:
            async for item in self.agent.stream(query, task.contextId, task.id):
                logger.info(f'Received item from stream: {type(item)}')
                
                # Agent to Agent call will return events,
                # Update the relevant ids to proxy back.
                if hasattr(item, 'root') and isinstance(
                    item.root, SendStreamingMessageSuccessResponse
                ):
                    logger.info('Processing SendStreamingMessageSuccessResponse')
                    event = item.root.result
                    if isinstance(
                        event,
                        (TaskStatusUpdateEvent | TaskArtifactUpdateEvent),
                    ):
                        logger.info(f'Enqueueing event: {event}')
                        await event_queue.enqueue_event(event)
                    continue
                
                # Handle different types of items that can be returned from the stream
                logger.info(f'Processing item type: {type(item)}')
                logger.info(f'info item: {item}')
                if isinstance(item, dict):
                    # If it's a dictionary, access it as before
                    logger.info('Item is a dictionary')
                    is_task_complete = item.get('is_task_complete', False)
                    require_user_input = item.get('require_user_input', False)
                    logger.info(f'is_task_complete: {is_task_complete}, require_user_input: {require_user_input}')
                    
                    if is_task_complete:
                        if item.get('response_type') == 'data':
                            part = DataPart(data=item.get('content', ''))
                            logger.info(f'Created DataPart')
                        else:
                            part = TextPart(text=item.get('content', ''))
                            logger.info(f'Created TextPart from content')
                    
                    if require_user_input:
                        logger.info('User input required, updating status')
                        await updater.update_status(
                            TaskState.input_required,
                            new_agent_text_message(
                                item['content'],
                                task.contextId,
                                task.id,
                            ),
                            final=True,
                        )
                        break
                    
                    # Update status for non-complete, non-input-required items
                    if not is_task_complete and not require_user_input:
                        await updater.update_status(
                            TaskState.working,
                            new_agent_text_message(
                                item['content'],
                                task.contextId,
                                task.id,
                            ),
                        )
                elif isinstance(item, TextPart):
                    # If it's already a TextPart, use it directly
                    logger.info(f'Item is a TextPart: {item.text[:50] if len(item.text) > 50 else item.text}')
                    is_task_complete = True  # Assume it's complete if we got a TextPart
                    part = item
                else:
                    # For other types, convert to string and create a TextPart
                    logger.info(f'Item is of type: {type(item)}')
                    is_task_complete = True
                    part = TextPart(text=str(item))
                    logger.info(f'Created TextPart from string: {str(item)[:50] if len(str(item)) > 50 else str(item)}')
                
                # For TextPart or other types, or dictionary items marked as complete
                if is_task_complete:
                    logger.info('Task is complete, adding artifact')
                    await updater.add_artifact(
                        [part],
                        name=f'{self.agent.agent_name}-result',
                    )
                    await updater.complete()
                    break
                
        except Exception as e:
            logger.error(f'Error in stream processing: {str(e)}')
            import traceback
            logger.error(f'Traceback: {traceback.format_exc()}')

    def _validate_request(self, context: RequestContext) -> bool:
        return False

    async def cancel(
        self, request: RequestContext, event_queue: EventQueue
    ) -> Task | None:
        raise ServerError(error=UnsupportedOperationError())
