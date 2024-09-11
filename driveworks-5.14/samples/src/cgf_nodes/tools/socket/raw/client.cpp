#include <iostream>
#include <sys/types.h>
#include <sys/socket.h>
#include <thread>
#include <chrono>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <signal.h>
#include <unistd.h>
#include <iostream>
#include <sstream>
#include <vector>
#include <cstring>
#include <string>
#include <fstream>


const int PORT = 8888;

void execute(int port, int size, int hz) {
    int clientSocket = socket(AF_INET, SOCK_STREAM, 0);
    if (clientSocket == -1) {
        std::cerr << "Error creating client socket." << std::endl;
        return;
    }

    sockaddr_in serverAddress;
    serverAddress.sin_family = AF_INET;
    serverAddress.sin_addr.s_addr = inet_addr("127.0.0.1");
    serverAddress.sin_port = htons(port);

    if (connect(clientSocket, reinterpret_cast<struct sockaddr*>(&serverAddress), sizeof(serverAddress)) == -1) {
        std::cerr << "Error connecting to server." << std::endl;
        close(clientSocket);
        return;
    }
    // usleep(hz);
    std::cout << "Connected to server on 127.0.0.1:" << port << ", size = " << size << ", hz = " << hz << std::endl;
    // usleep(10000000);
    char* buffer = new char[size + 1];
    memset(buffer, '1', size);
    buffer[size] = '\0';
    uint64_t index = 0;
    std::string file_name = std::to_string(port) + ".log";
    std::ofstream outfile(file_name, std::ios_base::app);
    if (!outfile) {
	std::cout << "open file fail" << std::endl;
    }
    while (true) {
        auto start = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
        ssize_t bytesWritten = write(clientSocket, buffer, strlen(buffer));
        auto end = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
        if (bytesWritten <= 0) {
            std::cerr << "Error sending message to server." << std::endl;
            break;
        }
        outfile << "** send No" << index <<  ", size: " << size << ", cost: "
            << (end - start)
            << " ms from port: " << port << std::endl;
        index++;
        // ssize_t bytesRead = read(clientSocket, buffer, sizeof(buffer));
        // if (bytesRead <= 0) {
        //     std::cerr << "Server disconnected." << std::endl;
        //     break;
        // }

        // std::cout << "Received from server: " << port << std::endl;
        usleep(1000*1000/hz);
    }

    close(clientSocket);
    delete buffer;
}

int main(int argc, char* argv[]) {
    int value = 0;
    int size = 0;
    int hz = 10;
    if (argc > 1) {
        value = std::atoi(argv[1]);
    }

    if (argc > 2) {
        size = std::atoi(argv[2]);
    }

    if (argc > 3) {
        hz = std::atoi(argv[3]);
    }

    std::cout << "base port : " << value << ", external size: " << 1024 + size << " with " << hz << std::endl;

    std::thread t0(execute, 18000 + value, 42047, 5);
    std::thread t1(execute, 18001 + value, 42047, 5);
    std::thread t2(execute, 18002 + value, 20282, 5);
    std::thread t3(execute, 18003 + value, 59434, 5);
    std::thread t4(execute, 18004 + value, 329, 100);
    std::thread t5(execute, 18005 + value, 74, 1);
    std::thread t6(execute, 18006 + value, 286800, 2);
    std::thread t7(execute, 18007 + value, 81, 50);
    std::thread t8(execute, 18008 + value, 98, 50);
    std::thread t9(execute, 18009 + value, 1104, 50);
    std::thread t10(execute, 18010 + value, 699, 50);
    std::thread t11(execute, 18011 + value, 334, 100);
    std::thread t12(execute, 18012 + value, 180, 10);
    std::thread t13(execute, 18013 + value, 280, 10);
    std::thread t14(execute, 18014 + value, 104, 20);
    std::thread t15(execute, 18015 + value, 68, 20);
    std::thread t16(execute, 18016 + value, 68, 20);
    std::thread t17(execute, 18017 + value, 286800, 20);
    std::thread t18(execute, 18018 + value, 138, 125);
    std::thread t19(execute, 18019 + value, 90, 50);
    std::thread t20(execute, 18020 + value, 90, 50);
    std::thread t21(execute, 18021 + value, 88, 50);
    std::thread t22(execute, 18022 + value, 124, 400);
    std::thread t23(execute, 18023 + value, 138, 100);
    std::thread t24(execute, 18024 + value, 334, 15);
    std::thread t25(execute, 18025 + value, 165, 30);
    std::thread t26(execute, 18026 + value, 74, 30);
    uint64_t base_size = 1024;
    std::thread t27(execute, 18027 + value, base_size + size, hz);

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