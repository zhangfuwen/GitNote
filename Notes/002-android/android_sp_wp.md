# Android 的强指针和弱指针

强指针根据这个对象的引用计数来决定要不要释放内存，当引用计数为0时释放内存。
但是强指针有一个问题是当两个对象互相持用强引用计数时，他们的引用计数没有办法变为0,因而不会被释放。解决办法是把其中一个变为弱指针。
某人持有某对象的弱指针时，这个对象的引用计数并不会+1。假如A持有对B的强指针，B持有对A的弱指针，则B的引用计数是1, A的引用计数是0。此时A可以被释放，A释放的同时，B的引用计数也降为0,B也得到了释放。

但是弱指针存在一个问题，就是持有弱指针的人要访问弱指针指向的对象的时候，怎么才能知道这个对象还在不在？
要解决这个问题，就要求弱指针也认识对象的引用计数，而且对象释放掉的情况下，这个引用计数仍然是存在的，即引用计数与对象本身具有不同的生命周期。
比如X 持有Y的弱指针，Y已经被释放掉了。X要访问Y的时候，要先检查一个Y的引用计数，如果引用计数是0,则表明Y释放掉了，不可以访问，如果引用计数不为0,那么可以访问，但访问之前，要先对引用计数+1,
防止使用过程中Y被释放掉。这个+1的过程是通过promote函数实现的。所谓promote函数，就是获取了Y的一个强指针。

听起来完美的解决了问题。

但是上面讲引用计数与对象要有不同的生命周期，那么保存引用计数的内存什么时候释放？由谁来释放？
这时，就要引入另一个变量，叫弱引用计数。上面我们讲的引用计数而改称为强引用计数。弱引用计数是用来释放引用计数所占用的内存的。当弱引用计数为0的时候，这块内存对释放。
弱引用计数本身也在这块内存里，与强引用计数具有相同的生命周期，所以不用怕再递归的需要引入第三个引用计数了。
弱引用计数什么时候变为0？答案是当最后一个弱指针析构的时候。

总结起来可以这样理解，强指针通过强引用计数维护对象的生命周期。弱指针通过弱引用计数维护强引用计数（和弱引用计数）的生命周期。

下面代码权当是**伪代码**：

```cpp
struct A { //普通类
    int a;
    int lot;
    int of;
    int members;
};

struct refs { //引用计数类
    int strongCount;
    int weakCount;
};

class StrongPointer<T> { // 强指针
    StrongPointer() {
        m_Ptr = new T();
        m_refs = new refs();
        m_refs.strongCount++;
        m_refs.weakCount++;
    }

    StrongPointer(WeakPointer<T> wp) { //弱指针promote为强指针时使用
        m_Ptr = wp.m_Ptr;
        m_refs = wp.m_refs;
        m_refs.strongCount++;
        m_refs.weakCount++;
    }

    ~StrongPointer() { //析构时释放对象内存
        m_refs.strongCount--;
        m_refs.weakCount--;
        if(m_refs.strongCount == 0) {
            delete m_Ptr;
        }
    }
    T * operator->() { //访问数据时用
        return m_Ptr;
    }
private:
    T * m_Ptr;
    refs * m_refs; 
};

class WeakPointer<T> {  // 弱指针类
    WeakPointer(StrongPointer sp) {
        m_refs = sp.m_refs;
        m_Ptr = sp.m_Ptr;
        m_refs.weakCount++;
    }

    ~WeakPointer() { //析构时释放引用计数对象的内存
        m_refs.weakCount--;
        if(weakCount==0) {
            delete m_refs;
        }
    }

    StrongPointer<T> promote() { //弱指针不能用来访问对象，要promote为强指针
        if(m_refs.strongCount == 0) {
            throw Execption.
        }
        return StrongPointer<T>(this);
    }

private:
    T * m_Ptr;
    refs * m_refs;

};
```