#include <iostream>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <cstring>
#include <thread>
#include <vector>
#include <chrono>

void execute(int port) {
    int serverSocket = socket(AF_INET, SOCK_STREAM, 0);
    if (serverSocket == -1) {
        std::cerr << "Error creating server socket." << std::endl;
        return;
    }

    sockaddr_in serverAddress;
    serverAddress.sin_family = AF_INET;
    serverAddress.sin_addr.s_addr = INADDR_ANY;
    serverAddress.sin_port = htons(port);

    if (bind(serverSocket, reinterpret_cast<struct sockaddr*>(&serverAddress), sizeof(serverAddress)) == -1) {
        std::cerr << "Error binding server socket." << std::endl;
        close(serverSocket);
        return;
    }

    if (listen(serverSocket, 10) == -1) {
        std::cerr << "Error listening on server socket." << std::endl;
        close(serverSocket);
        return;
    }

    std::cout << "Server listening on 127.0.0.1:" << port << std::endl;

    int clientSocket = accept(serverSocket, nullptr, nullptr);
    if (clientSocket == -1) {
        std::cerr << "Error accepting client connection." << std::endl;
        close(serverSocket);
        return;
    }

    char buffer[1024];
    char ret[1] = {'1'};
    uint64_t index = 0;
    while (true) {
        ssize_t bytesRead = read(clientSocket, buffer, sizeof(buffer));
        if (bytesRead <= 0) {
            std::cerr << "Client disconnected." << std::endl;
            break;
        }

        std::cout << "Received No" << index << ", cost: "
            << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count()
            << " ms from port: " << port << std::endl;
        ++index;

        // ssize_t bytesWritten = write(clientSocket, ret, bytesRead);
        // if (bytesWritten <= 0) {
        //     std::cerr << "Error sending response to client." << std::endl;
        //     break;
        // }
    }

    close(clientSocket);
    close(serverSocket);
}

int main(int argc, char* argv[]) {
    int value = 0;
    if (argc > 1) {
      value = std::atoi(argv[1]);
    }
    std::cout << "base port : " << value << std::endl;

    std::thread t0(execute, 18000 + value);
    std::thread t1(execute, 18001 + value);
    std::thread t2(execute, 18002 + value);
    std::thread t3(execute, 18003 + value);
    std::thread t4(execute, 18004 + value);
    std::thread t5(execute, 18005 + value);
    std::thread t6(execute, 18006 + value);
    std::thread t7(execute, 18007 + value);
    std::thread t8(execute, 18008 + value);
    std::thread t9(execute, 18009 + value);
    std::thread t10(execute, 18010 + value);
    std::thread t11(execute, 18011 + value);
    std::thread t12(execute, 18012 + value);
    std::thread t13(execute, 18013 + value);
    std::thread t14(execute, 18014 + value);
    std::thread t15(execute, 18015 + value);
    std::thread t16(execute, 18016 + value);
    std::thread t17(execute, 18017 + value);
    std::thread t18(execute, 18018 + value);
    std::thread t19(execute, 18019 + value);
    std::thread t20(execute, 18020 + value);
    std::thread t21(execute, 18021 + value);
    std::thread t22(execute, 18022 + value);
    std::thread t23(execute, 18023 + value);
    std::thread t24(execute, 18024 + value);
    std::thread t25(execute, 18025 + value);
    std::thread t26(execute, 18026 + value);
    std::thread t27(execute, 18027 + value);

    t0.join();
    t1.join();
    t2.join();
    t3.join();
    t4.join();
    t5.join();
    t6.join();
    t7.join();
    t8.join();
    t9.join();
    t10.join();
    t11.join();
    t12.join();
    t13.join();
    t14.join();
    t15.join();
    t16.join();
    t17.join();
    t18.join();
    t19.join();
    t20.join();
    t21.join();
    t22.join();
    t23.join();
    t24.join();
    t25.join();
    t26.join();
    t27.join();

    return 0;
}