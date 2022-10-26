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

# C++ type list

```cpp
using MyTypeList = std::tuple<GenericFile,
                                  TextFile,
                                  ImageFile,
                                  AudioFile,
                                  VideoFile>;
                                  
template<typename List, int... ints>
void UseTypes(std::integer_sequence<int, ints...> int_seq)
{
    ((printf("%d\n",ints)), ...);
    ([&]{
         printf("%d\n", ints);
         using E = typename std::tuple_element<ints, List>::type;
         auto name = E::name(); // call static member function
         auto ptr = std::make_shared<E>(); // construct an instance
         auto instName = ptr->instName(); // call instance member function
     }(),...); // fold expression
}

const int numTypes = std::tuple_size_v<MyTypeList>;
UseTypes<MyTypeList>(std::make_integer_sequence<int, numTypes>{});
```

# C++ print variant

```cpp
using Error = int;
using File = std::variant<
    Error,
    std::shared_ptr<GenericFile>,
      std::shared_ptr<TextFile>,
      std::shared_ptr<ImageFile>,
      std::shared_ptr<AudioFile>,
      std::shared_ptr<VideoFile>>;


template <typename Variant, int... ints>
void PrintResult(Variant & v, std::integer_sequence<int, ints...>) {
    ([&]{
        printf("%d\n", ints);
        using E = typename std::variant_alternative_t<ints, Variant>;
        if constexpr (std::is_same_v<E, int>) {
            if(std::holds_alternative<E>(v) { // runtime type test
                std::cout << "error " << std::get<int>(v) << std::endl;
            }
        } else {
            if(std::holds_alternative<E>(v) { // runtime type test
                std::cout << std::get<E>(v)->toString() << std::endl; // other types must implement toString
            }
        }
    }(),...);
}

auto ret = SomeFunctionThatReturnsVariant();
constexpr int numTypes = std::variant_size_v<FileInfo>;
PrintResult(ret, std::make_integer_sequence<int, numTypes>{});

```

# static member function and compile time polymorphism

just return variant and use printResult like in the previous section. 

other  useful templates:

## std::get_if

run time test, pass type or zero-based index, returns a pointer or nullptr.

```cpp
std::variant<int, float> v;
int *pv = std::get_if<Type>(&v);
if(pv != nullptr) {
    printf("%d\n", *pv);
}
float * pf = std::get_if<1>(&v);
if(pf != nullptr) {
    printf("%f\n", *pf);
}
```

## std::visit

```cpp
include <iomanip>
#include <iostream>
#include <string>
#include <type_traits>
#include <variant>
#include <vector>
 
// the variant to visit
using var_t = std::variant<int, long, double, std::string>;
 
// helper constant for the visitor #3
template<class> inline constexpr bool always_false_v = false;
 
// helper type for the visitor #4
template<class... Ts> struct overloaded : Ts... { using Ts::operator()...; };
// explicit deduction guide (not needed as of C++20)
template<class... Ts> overloaded(Ts...) -> overloaded<Ts...>;
 
int main() {
    std::vector<var_t> vec = {10, 15l, 1.5, "hello"};
    for (auto& v: vec) {
 
        // 1. void visitor, only called for side-effects (here, for I/O)
        std::visit([](auto&& arg){std::cout << arg;}, v);
 
        // 2. value-returning visitor, demonstrates the idiom of returning another variant
        var_t w = std::visit([](auto&& arg) -> var_t {return arg + arg;}, v);
 
        // 3. type-matching visitor: a lambda that handles each type differently
        std::cout << ". After doubling, variant holds ";
        std::visit([](auto&& arg) {
            using T = std::decay_t<decltype(arg)>;
            if constexpr (std::is_same_v<T, int>)
                std::cout << "int with value " << arg << '\n';
            else if constexpr (std::is_same_v<T, long>)
                std::cout << "long with value " << arg << '\n';
            else if constexpr (std::is_same_v<T, double>)
                std::cout << "double with value " << arg << '\n';
            else if constexpr (std::is_same_v<T, std::string>)
                std::cout << "std::string with value " << std::quoted(arg) << '\n';
            else 
                static_assert(always_false_v<T>, "non-exhaustive visitor!");
        }, w);
    }
 
    for (auto& v: vec) {
        // 4. another type-matching visitor: a class with 3 overloaded operator()'s
        // Note: The `(auto arg)` template operator() will bind to `int` and `long`
        //       in this case, but in its absence the `(double arg)` operator()
        //       *will also* bind to `int` and `long` because both are implicitly
        //       convertible to double. When using this form, care has to be taken
        //       that implicit conversions are handled correctly.
        std::visit(overloaded {
            [](auto arg) { std::cout << arg << ' '; },
            [](double arg) { std::cout << std::fixed << arg << ' '; },
            [](const std::string& arg) { std::cout << std::quoted(arg) << ' '; }
        }, v);
    }
}
```

## simplified PrintVariant 

```cpp
void PrintInfo(const Variant &info) {
    std::visit([](auto && arg) {
        using T= typename std::decay_t<decltype(arg)>;
        if constexpr (std::is_same_v<Error, T>) {
            FUN_INFO("error %d", arg);
        }  else {
            FUN_INFO("result: %s", arg->toString.c_str());
        }
    }, info);
}
```

