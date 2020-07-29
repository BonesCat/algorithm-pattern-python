# 排序

## 常考排序

### 快速排序

```Python
import random

def partition(nums, left, right):
    if left >= right:
        return

    pivot_idx = random.randint(left, right)
    pivot = nums[pivot_idx]
    
    nums[right], nums[pivot_idx] = nums[pivot_idx], nums[right]
            
    partition_idx = left
    for i in range(left, right):
        if nums[i] < pivot:
            nums[partition_idx], nums[i] = nums[i], nums[partition_idx]
            partition_idx += 1
            
    nums[right], nums[partition_idx] = nums[partition_idx], nums[right]

    partition(nums, partition_idx + 1, right)
    partition(nums, left, partition_idx - 1)

    return

def quicksort(A):
    partition(A, 0, len(A) - 1)
    return A

if __name__ == '__main__':
    a = [7, 6, 8, 5, 2, 1, 3, 4, 0, 9, 10]
    print(a)
    print(quicksort(a))
```

### 归并排序

```Python
def merge(A, B):
    C = []
    i, j = 0, 0
    while i < len(A) and j < len(B):
        if A[i] <= B[j]:
            C.append(A[i])
            i += 1
        else:
            C.append(B[j])
            j += 1
    
    if i < len(A):
        C += A[i:]
    
    if j < len(B):
        C += B[j:]
    
    return C

def mergsort(A):
    n = len(A)
    if n < 2:
        return A[:]
    
    left = mergsort(A[:n // 2])
    right = mergsort(A[n // 2:])

    return merge(left, right)

if __name__ == '__main__':
    a = [7, 6, 8, 5, 2, 1, 3, 4, 0, 9, 10]
    print(a)
    print(mergsort(a))
```

### 堆排序

用数组表示的完美二叉树 complete binary tree

> 完美二叉树 VS 其他二叉树

![image.png](https://img.fuiboom.com/img/tree_type.png)

[动画展示](https://www.bilibili.com/video/av18980178/)

![image.png](https://img.fuiboom.com/img/heap.png)

核心代码

```Python
def heap_adjust(A, start=0, end=None):
    if end is None:
        end = len(A)
    
    while start is not None and start < end // 2:
        l, r = start * 2 + 1, start * 2 + 2
        swap = None

        if A[l] > A[start]:
            swap = l
        if r < end and A[r] > A[start] and (swap is None or A[r] > A[l]):
            swap = r

        if swap is not None:
            A[start], A[swap] = A[swap], A[start]
            
        start = swap
    
    return

def heapsort(A):

    # construct max heap
    n = len(A)
    for i in range(n // 2 - 1, -1, -1):
        heap_adjust(A, i)
    
    # sort
    for i in range(n - 1, 0, -1):
        A[0], A[i] = A[i], A[0]
        heap_adjust(A, end=i)
    
    return A

# test
if __name__ == '__main__':
    a = [7, 6, 8, 5, 2, 1, 3, 4, 0, 9, 10]
    print(a)
    print(heapsort(a))
```

## 题目

### [kth-largest-element-in-an-array](https://leetcode-cn.com/problems/kth-largest-element-in-an-array/)

思路 1: sort 后取第 k 个，最简单直接，复杂度 O(N log N) 代码：

```Python
class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        # 不适用其余技巧，先系统自带排序（升序）
        # 然后返回第k大
        # 返回最终排定以后位于 len - k 的那个元素
        if len(nums) == 0:
            return 0
        nums = sorted(nums)
        return nums[-k]
```        

思路 2: 使用最小堆，复杂度 O(N log k)

```Python
class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        # note that in practice there is a more efficient python build-in function heapq.nlargest(k, nums)
        min_heap = []
        
        for num in nums:
            if len(min_heap) < k:
                heapq.heappush(min_heap, num)
            else:
                if num > min_heap[0]:
                    heapq.heappushpop(min_heap, num)
        
        return min_heap[0]

import heapq
class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        # 本题本质是寻找topK大的数
        # 先根据nums的前k个数建立小顶堆，
        # 后续的每个数，先进行判断
        # 若当前num大于堆顶，则堆顶出，进去一个数
        # 这样最后这个小顶堆里保存的都是大数
        # 堆顶即为第k大的元素

        if len(nums) == 0 or k == 0:
            return 0
        # list定义优先队列
        queue = []

        # 建立K个值的小顶堆
        for i in range(k):
            heapq.heappush(queue,nums[i])
        
        # 遍历nums后续元素
        for i in range(k, len(nums)):
            # 若nums元素大于小堆顶
            # 则出堆一个，入堆新元素
            if nums[i] > queue[0]:
                heapq.heappop(queue)
                heapq.heappush(queue, nums[i])
        
        # 结束之后，小顶堆保存k个nums中的最大数
        # 堆顶为第k大
        return queue[0]
```

思路 3: Quick select，方式类似于快排，每次 partition 后检查 pivot 是否为第 k 个元素，如果是则直接返回，如果比 k 大，则继续 partition 小于 pivot 的元素，如果比 k 小则继续 partition 大于 pivot 的元素。相较于快排，quick select 每次只需 partition 一侧，因此平均复杂度为 O(N)

```Python
class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        
        k -= 1 # 0-based index
        
        def partition(left, right):
            pivot_idx = random.randint(left, right)
            pivot = nums[pivot_idx]
            
            nums[right], nums[pivot_idx] = nums[pivot_idx], nums[right]
            
            partition_idx = left
            for i in range(left, right):
                if nums[i] > pivot:
                    nums[partition_idx], nums[i] = nums[i], nums[partition_idx]
                    partition_idx += 1
            
            nums[right], nums[partition_idx] = nums[partition_idx], nums[right]
            
            return partition_idx
        
        left, right = 0, len(nums) - 1
        while True:
            partition_idx = partition(left, right)
            if partition_idx == k:
                return nums[k]
            elif partition_idx < k:
                left = partition_idx + 1
            else:
                right = partition_idx - 1
                 
```
思路4：使用原始的快排方法
```Python
class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        def quick_sort(list1, start, end):
            # 递归退出条件
            if start >= end:
                return
            
            # 将起始元素设置为要寻找位置的基准元素pivot
            pivot = list1[start]
            # left为序列左边的由左向右移动的游标
            left = start
            # left为序列右边的由右向左移动的游标
            right = end
            # 此循环以pivot为分界，将数据按照大小分为
            # 左边都小于pivot, 右边大于等于pivot
            while left < right:
                # 当left和right未重合，而且right指向元素的值大于等于基准元素值pivot
                # right向左移动
                while left < right and list1[right] >= pivot:
                    right -= 1
                # 将right指向的元素放到left上
                list1[left] = list1[right]
                
                # 当left和right未重合，而且left指向元素的值小于基准元素值pivot
                # left不断右移
                while left < right and list1[left] < pivot:
                    left += 1
                # 将left指向的元素放到right上
                list1[right] = list1[left]
            # 循环结束后，right和left重合，此时所指位置为基准元素的正确位置
            # 将基准元素放到该位置
            list1[left] = pivot
            
            # 使用递归分别对剩下的两部分，进行快速排序
            quick_sort(list1, start, left-1)
            quick_sort(list1, left+1, end)
        quick_sort(nums,0, len(nums)-1 )
        return nums[-k]
```


## 参考

[十大经典排序](https://www.cnblogs.com/onepixel/p/7674659.html)

[二叉堆](https://labuladong.gitbook.io/algo/shu-ju-jie-gou-xi-lie/er-cha-dui-xiang-jie-shi-xian-you-xian-ji-dui-lie)

## 练习

- [ ] 手写快排、归并、堆排序
