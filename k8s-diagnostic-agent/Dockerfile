# Use a minimal base image
FROM alpine:3.18

# Install necessary packages
RUN apk add --no-cache tcpdump iputils iproute2 curl bind-tools net-tools procps strace lsof iperf3

# Set a default command to keep the container running
CMD ["tail", "-f", "/dev/null"]