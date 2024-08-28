/*
    execute this app with cmd:
    LD_PRELOAD=[your target path]/aarch64/install/bin/common_cgf_channel/libnodelay.so  [your target path]aarch64/install/bin/common_cgf_channel/tcp_nodelay_preload_test

    or
    export LD_PRELOAD=[your target path]/aarch64/install/bin/common_cgf_channel/libnodelay.so
    [your target path]aarch64/install/bin/common_cgf_channel/tcp_nodelay_preload_test
*/

#include <stdio.h>
#include <unistd.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <netinet/tcp.h>
#include <string.h>
#include <errno.h>

int main() {
    const int sockfd = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    int optval;
    socklen_t len = sizeof(optval);
    if (sockfd < 0) {
        fprintf(stderr, "socket() failed: %s\n", strerror(errno));
        return 1;
    }

    if (getsockopt(sockfd, IPPROTO_TCP, TCP_NODELAY, &optval, &len) != 0) {
        fprintf(stderr, "getsockopt() failed: %s\n", strerror(errno));
        close(sockfd);
        return 1;
    }

    close(sockfd);
    if (optval) {
        fprintf(stdout, "TCP_NODELAY is set.\n");
        return 0;
    } 
    fprintf(stderr, "TCP_NODELAY is NOT set.\n");
    return 1;
}
