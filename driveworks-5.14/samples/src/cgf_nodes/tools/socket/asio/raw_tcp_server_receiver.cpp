#include <thread>
#include <chrono>
#include <vector>
#include <signal.h>
#include <boost/asio.hpp>
#include <boost/thread.hpp>

namespace
{
bool keepGoing = true;
void shutdown(int)
{
  keepGoing = false;
}

std::size_t bytesAccum = 0;
void justReceive(boost::system::error_code ec, std::size_t bytesReceived,
    boost::asio::ip::tcp::socket &socket, std::vector<unsigned char> &buffer)
{
  bytesAccum += bytesReceived;

  auto end = buffer.begin() + bytesReceived;
  for (auto it = buffer.begin(); it != end; ++it)
  {
    if (*it == 'e')
    {
      std::printf("server got: %lu\n", bytesAccum);
      bytesAccum = 0;
    }
  }

  socket.async_receive(
      boost::asio::buffer(buffer, 2048),
      0,
      boost::bind(justReceive, _1, _2, boost::ref(socket),
                                       boost::ref(buffer)));
}
}

int main(int, char **)
{
  signal(SIGINT, shutdown);

  boost::asio::io_service io;
  boost::asio::io_service::work work(io);

  boost::thread t1(boost::bind(&boost::asio::io_service::run, &io));
  boost::thread t2(boost::bind(&boost::asio::io_service::run, &io));
  boost::thread t3(boost::bind(&boost::asio::io_service::run, &io));
  boost::thread t4(boost::bind(&boost::asio::io_service::run, &io));

  boost::asio::ip::tcp::acceptor acceptor(io,
      boost::asio::ip::tcp::endpoint(
          boost::asio::ip::address::from_string("127.0.0.1"), 18000));
  boost::asio::ip::tcp::socket socket(io);

  // accept 1 client
  std::vector<unsigned char> buffer(2048, 0);
  acceptor.async_accept(socket, [&socket, &buffer](boost::system::error_code ec)
  {
      // options
      socket.set_option(boost::asio::ip::tcp::no_delay(true));
      socket.set_option(boost::asio::socket_base::receive_buffer_size(1920 * 1080 * 4));
      socket.set_option(boost::asio::socket_base::send_buffer_size(1920 * 1080 * 4));

      socket.async_receive(
          boost::asio::buffer(buffer, 2048),
          0,
          boost::bind(justReceive, _1, _2, boost::ref(socket),
                                           boost::ref(buffer)));
  });

  while (keepGoing)
  {
    std::this_thread::sleep_for(std::chrono::seconds(1));
  }

  io.stop();

  t1.join();
  t2.join();
  t3.join();
  t4.join();

  std::printf("server: goodbye\n");
}