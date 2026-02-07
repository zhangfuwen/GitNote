---

title: 快排算法

---

# 基本思路

我觉得可以称之为`安放算法`.

快排的第一步操作是分区partition，但其实分区的基本方法是`安放`.
随机选一个数，把它安放到它应有的位置。

比如说数为9 8 7 6 5 4 3 2 1 

我随机选一个数，比如说6，那它的位置一定是放在第5位。我怎么才能把一个选择的数(pivot)放到应有的位置呢？
方法就是便历一次所有的数，把比他小的放在它前面，把比它大的放在它后面。

结果就是：

[任意顺序 5 4 3 2 1] **6** [任意顺序 6 7 8 9]

这里涉及两个问题，一个是随机的数怎么选，最简单的办法是选第0个数，也可以选中间的数，可能选中间的数更好一些。
如果选的是中间的数，我们要把它和第0个数交换下位置，这样问题又变成了选择第0个数做为pivot的算法了。

第二个问题是怎么安放，一个简单的办法是引入第一个数据，然后遍历第一个数组，把比arr[pivot]小的数先放进去，然后放pivot，然后放比它大的数。

这样浪费了一点点空间，其实问题也不大。

```cpp
int partition(vector<int> &arr, int left, int right) {
    vector<int> tmp;
    int pivot = left;
    // enqueue smaller numbers
    for(int i = left+1; i<=right; i++) {
       if(arr[i] < arr[pivot])  {
           tmp.push_back(arr[i]);
       }
    }
    // enqueue pivot
    int ret = tmp.size();
    tmp.push_back(arr[pivot]);
    // enqueue large numbers
    for(int i = left+1; i<=right; i++) {
       if(arr[i] > arr[pivot])  {
           tmp.push_back(arr[i]);
       }
    }
    return ret;
}
```

经过partition之后，数组被分成两部分，左右两部分都是未排序的状态，我们就可以采用分治了：

```cpp
void mysort(vector<int> &arr, int left, int right) {
    if(left >= right) {
        return;
    }
    int pivot = partition(arr, left, right);
    mysort(arr, left, pivot - 1);
    mysort(arr, pivot +1, right);
}

int main() {
    mysort(arr, 0, arr.size() -1);
    return 0;
}

```

# partition算法的优化

partition的第一个写法超级低效，其实完全可以原地实现`安放`的，而且有多种方法.

```cpp
int partition(vector<int> &arr, int left, int right) {
    int pivot = left;
    int index = left + 1; // room for next smaller number
    for(int i = index; i <= right; i++) {
        if(arr[i] < arr[pivot]) { // found a smaller number
            std::swap(arr[index], arr[i]); // 把小数放前面去，index位置原来的不知道是什么的数换后面去
            // 这里i一定是大于等于index的，所以i一定在index后面
            index++; // index指向下一个被牺牲用于存放小数的位置
        }
    }
    return index-1;
}
```