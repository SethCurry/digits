
#include <chrono>
#include <iostream>
#include "Leap.h"
#include "./include/libdigitspy.hpp"
#include "libdigits.hpp"

using namespace Leap;

namespace sc = std::chrono;

static LibdigitspyListener listener;
static Controller controller;


extern "C" {
    void start_listener(int interval, int duration) {
        // Create a sample listener and controller
        listener.setInterval(interval);
        listener.setBufferSize(duration);

        // Have the sample listener receive events from the controller
        controller.addListener(listener);
    }

    void get_frames(float **dest) {
        listener.getFrames(dest);
    }
}



void LibdigitspyListener::setInterval(long newInterval) {
    interval = newInterval;
}

void LibdigitspyListener::setBufferSize(int newBufferSize) {
    bufferSize = newBufferSize;
}

void LibdigitspyListener::onInit(const Controller& controller) {
  std::cerr << "Initialized" << std::endl;
}

void LibdigitspyListener::onConnect(const Controller& controller) {
  std::cerr << "Connected" << std::endl;
}

void LibdigitspyListener::onDisconnect(const Controller& controller) {
  // Note: not dispatched when running in a debugger.
  std::cerr << "Disconnected" << std::endl;
}

void LibdigitspyListener::onExit(const Controller& controller) {
  std::cerr << "Exited" << std::endl;
}

void LibdigitspyListener::onFrame(const Controller &controller) {
    auto time = sc::system_clock::now(); // get the current time
    auto since_epoch = time.time_since_epoch(); // get the duration since epoch

    // I don't know what system_clock returns
    // I think it's uint64_t nanoseconds since epoch
    // Either way this duration_cast will do the right thing
    auto millis = sc::duration_cast<sc::milliseconds>(since_epoch);

    long now = millis.count(); // just like java (new Date()).getTime();

    if (now >= lastEmit + interval) {
        const Frame frame = controller.frame();
          if (frame.hands().count() < 1 || frame.hands().count() > 1) {
            return;
          }

          // Emit one comma-separated line of model inputs per hand.
          HandList hands = frame.hands();

          for (HandList::const_iterator hl = hands.begin(); hl != hands.end(); ++hl) {
            const Hand hand = *hl;
            std::vector<float> inputs = modelInputs(hand);

            std::lock_guard<std::mutex> guard(bufferedFramesMut);

            auto pos = bufferedFrames.begin();
            bufferedFrames.insert(pos, inputs);

            while (bufferedFrames.size() > bufferSize) {
                bufferedFrames.pop_back();
            }
          };
          lastEmit = now;
        };
    };

void LibdigitspyListener::getFrames(float **into) {
    for (size_t i = 0; i < listener.size; i++) {
        for (int j = 0; j < 66; j++) {
            into[i][j] = bufferedFrames.at(i).at(j);
        }
    }
}

void LibdigitspyListener::onFocusGained(const Controller& controller) {
  std::cerr << "Focus Gained" << std::endl;
}

void LibdigitspyListener::onFocusLost(const Controller& controller) {
  std::cerr << "Focus Lost" << std::endl;
}

void LibdigitspyListener::onDeviceChange(const Controller& controller) {
  std::cout << "Device Changed" << std::endl;
  const DeviceList devices = controller.devices();

  for (int i = 0; i < devices.count(); ++i) {
    std::cerr << "id: " << devices[i].toString() << std::endl;
    std::cerr << "  isStreaming: " << (devices[i].isStreaming() ? "true" : "false") << std::endl;
  }
}

void LibdigitspyListener::onServiceConnect(const Controller& controller) {
  std::cerr << "Service Connected" << std::endl;
}

void LibdigitspyListener::onServiceDisconnect(const Controller& controller) {
  std::cerr << "Service Disconnected" << std::endl;
}
