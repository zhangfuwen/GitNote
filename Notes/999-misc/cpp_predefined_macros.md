---

title: cpp 

---

# predefined macros

## OS Type

```
    __linux__       Defined on Linux
    __sun           Defined on Solaris
    __FreeBSD__     Defined on FreeBSD
    __NetBSD__      Defined on NetBSD
    __OpenBSD__     Defined on OpenBSD
    __APPLE__       Defined on Mac OS X
    __hpux          Defined on HP-UX
    __osf__         Defined on Tru64 UNIX (formerly DEC OSF1)
    __sgi           Defined on Irix
    _AIX            Defined on AIX
    _WIN32          Defined on Windows
```

usage:

```cpp
#ifdef __linux__ 
    //linux code goes here
#elif _WIN32
    // windows code goes here
#else

#endif
```


# std::bind/std::thread pass by reference

std::bind/std::thread传参数时会强制拷贝，所以传引用不会起做用，比如说：

```cpp

int main() {
    int val = 5;
    std::thread thr([](int & param){
        param = 6;
    }, val);
    thr.join();
    printf("%d\n", val); // result is 5 instead of 6.
    return 0;
}
```

由于val在传参数拷贝了一份，在线程内修改param，并不影响val的值，所以线程退出后val的值还是5而不是6。

要解决这个问题可以用std::ref(val)这种方式去传参。std::ref(val)会把val包装在一个对象里，这个对象可以像int &一样使用（隐式转换）, 但是copy的时候依然保持引用状态。下面的代码是可以达到在线程里修改val值的目的的。

```cpp
int main() {
    int val = 5;
    std::thread thr([](int & param){
        param = 6;
    }, std::ref(val));
    thr.join();
    printf("%d\n", val); // result is 6
    return 0;
}
```

# std::bind/std::thread pass by reference完美转发

std::bind/std::thread无法用std::forward实现完美转发了，需要一个helper方法：

```cpp
    template <class T> std::reference_wrapper<T> maybe_wrap(T& val) { return std::ref(val); }
    template <class T> T&& maybe_wrap(T&& val) { return std::forward<T>(val); }
    
    // 转发时
    auto taskPtr = std::make_shared<packaged_task_type>(std::bind(
            std::forward<std::function<Ret(Args...)>>(callable), maybe_wrap(std::forward<Args>(args))...));
    // 即
    maybe_wrap(std::forward<Args>(args))...
```    


