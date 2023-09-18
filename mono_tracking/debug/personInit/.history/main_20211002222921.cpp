// TOPIC: Introduction to thread in c++ (c++11)

// QUESTIONS
// 1. What do you understand by thread and give one example in C++?

// ANSWER
// 0. In every application there is a default thread which is main(), in side this we create other threads.
// 1. A thread is also known as lightweight process. Idea is achieve parallelism by dividing a process into multiple threads. 
//    For example:
//    (a) The browser has multiple tabs that can be different threads. 
//    (b) MS Word must be using multiple threads, one thread to format the text, another thread to process inputs (spell checker)
//    (c) Visual Studio code editor would be using threading for auto completing the code. (Intellicence)

// WAYS TO CREATE THREADS IN C++11
// 1. Function Pointers
// 2. Lambda Functions
// 3. Functors
// 4. Member Functions
// 5. Static Member functions

#include<iostream>
#include<thread>
#include<chrono>
#include<algorithm>
using namespace std;
using namespace std::chrono;
typedef unsigned long long ull;

ull OddSum = 0;
ull EvenSum = 0;

void findEven(ull _start, ull _end)
{
    for (ull i = _start; i < _end; ++i)
    {
        if((i & 1) == 0)
            EvenSum += i;
    }
}
void findOdd(ull _start, ull _end)
{
    for (ull i = _start; i < _end; ++i)
    {
        if((i & 1) == 1)
            OddSum += i;
    }
}
int main()
{
    ull start = 0, end = 1900000000;
    auto startTime = high_resolution_clock::now();

    std::thread t1(findEven, start, end);  // 启动线程
    std::thread t2(findOdd, start, end);  // 启动线程

    t1.join();
    cout << "EvenSum: " << EvenSum << endl;
    cout << "OddSum:  " << OddSum << endl << endl;
    t2.join();

    // findOdd(start, end);
    // findEven(start, end);
    auto endTime = high_resolution_clock::now();
    duration<double> duration = endTime - startTime;
    cout << "OddSum:  " << OddSum << endl;
    cout << "EvenSum: " << EvenSum << endl;
    cout << "Cost:    " << duration.count() << "s" << endl;
    return 0;
}