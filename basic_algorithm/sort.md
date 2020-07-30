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
### [最小的k个数](https://leetcode-cn.com/problems/zui-xiao-de-kge-shu-lcof/)

思路很多，可以帮助梳理基础的快排，还有堆排序，特别注意Python利用生成大顶堆的方法（加负号），注意负值大小关系的处理

还有记住：MaxTopK:用小顶堆 MinTopK：用大顶堆

```Python
import heapq
class Solution:
    def getLeastNumbers(self, arr: List[int], k: int) -> List[int]:
        # 使用大根堆，然后比顶小的数据进入
        # 堆顶数据不断剔除，让堆一直维持在k大小
        
        if len(arr) == 0 or k == 0:
            return []
        queue = []
        res = []
        for i in range(k):
            # Python 小根堆加负号，变成大根堆
            heapq.heappush(queue, -arr[i])
        
        # 维持大小为k，将arr后续元素比较后入堆
        for i in range(k, len(arr)):
            # 若当前arr元素小于堆顶元素
            # 则堆顶元素出堆，arr【i】入堆
            # 因为取反了，所以是大于堆顶的入堆
            if (-arr[i]) > queue[0]:
                heapq.heappop(queue)
                heapq.heappush(queue, (-arr[i]))
        for num in queue:
            res.append(-num)
                       
        return res
        
class Solution:
    def getLeastNumbers(self, arr: List[int], k: int) -> List[int]:
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

        quick_sort(arr, 0, len(arr)-1)
        return arr[:k]        
        
```        

### [sortArray](https://leetcode-cn.com/problems/sort-an-array/)

参考了大佬的讲解，还有代码，主要为了熟悉基础的排序，还有其他的一些排序没有写进去
详情见:
###[Python实现十大经典排序](https://leetcode-cn.com/problems/sort-an-array/solution/python-shi-xian-de-shi-da-jing-dian-pai-xu-suan-fa/)
```
Python

class Solution:
    def sortArray(self, nums: List[int]) -> List[int]:
        # 1.使用传统的选择排序（升序）,时间复杂度为O(n^2),超时
        # 两个for循环，
        # 外层for负责定位当前需要确定的位置i
        # 内层for负责进行比较，找到比i小的j的位置
        n = len(nums)
        for i in range(n):
            # 内层比较的数据为i到(n-1)
            for j in range(i, n):
                # 若当前i比j位置元素大，则交换两者
                if nums[i] > nums[j]:
                    # 交换两者
                    nums[i], nums[j] = nums[j], nums[i]
        return nums
        
class Solution:
    def sortArray(self, nums: List[int]) -> List[int]:
        # 2.使用传统的冒泡排序（升序）
        # 两个for循环，
        # 外层for负责定位当前需要确定的位置i
        # 内层for负责进行两两比较，不断将大数沉底至右边
        n = len(nums)
        for i in range(n):
            # 内层两两比较的始末范围为1-(n-i)
            # 右边开始的i的都已经确定
            # 还剩下0-（n-i）位置的没有确定
            # 需要进行两两比较，所以从1开始
            for j in range(1, n-i):
                # 若当前j -1比j位置元素大，则交换两者
                if nums[j -1] > nums[j]:
                    # 交换两者
                    nums[j -1], nums[j] = nums[j], nums[j -1]
        return nums    

class Solution:
    def sortArray(self, nums: List[int]) -> List[int]:
        # 3.使用传统的插入排序（升序）
        # 外层一个for循环，负责定位当前i的位置，i属于1~n-1
        # 内层while，负责从当前i向左，不断两两比较
        # 每次将大的向后移动
        n = len(nums)
        for i in range(n):
            # 内层两两比较的始末范围为1~i
            # 需要进行两两比较，所以从1开始
            # 每次比较完成之后，i进行自减
            # 内层while保证i向左都是有序
            # 若i大于0且当前i -1比i位置元素大，则交换两者
            while i > 0 and nums[i -1] > nums[i]:
                nums[i -1], nums[i] = nums[i], nums[i -1]
                i -= 1
        return nums    

class Solution:
    def sortArray(self, nums: List[int]) -> List[int]:
        # 4.使用归并排序（升序）
        '''
        把长度为n的输入序列分成长度 n/2的子序列；
        不断递归切分至只有一个元素的子序列，
        系统自动开始向上回溯。
        然后对所有两两组队的子序列采用归并排序；
        既两个子序列，每次各出一个数比较排序，
        合并所有子序列。
        '''
        
        # 若数组长度为0或1，直接返回数组
        if len(nums) <= 1:
            return nums
        # devide
        mid = len(nums) // 2
        # 不断递归，切分 
        left = self.sortArray(nums[:mid])
        right = self.sortArray(nums[mid:]) 

        # merge
        return self.merge(left, right)

    def merge(self, left, right):
        # 使用额外空间
        res = []
        i, j = 0, 0
        while (i < len(left) and j < len(right)):
            # 若left 小于 当前right，则小的进入res
            # i右移动1
            if left[i] <= right[j]:
                res.append(left[i])
                i += 1
            # 若right 小于 当前left，则小的进入res
            else:
                res.append(right[j])
                j += 1
        # 若循环结束，left和right还有剩余，则直接加入到结果res中
        res += left[i:]
        res += right[j:]
        return res

class Solution:
    def sortArray(self, nums: List[int]) -> List[int]:
        # 4.使用计数排序（升序），算法时间复杂度O（n）
        '''
        这个方法对数据要求很高，如果我数据范围太大而且十分稀疏就不太好用了。
        比较适合排数据范围小但是每个数据点有很多个的那种数据
        本题因为数据元素，其速度小于快排
        '''
        # 思想：找到nums中的最大_max和最小数据_min
        # 并以_max和_min为范围大小建立数组tmp_arr
        # tmp_arr的下标是对应nums中的数据值减_min
        # 遍历原始nums数据，并num出现的次数，累加到tmp_arr[num-_min]中去
        # 然后遍历tmp_arr，还原数据

        # 特殊值判断
        if not nums:
            return []

        n = len(nums)

        # 得到数组的最值
        min_d, max_d = min(nums), max(nums)
        # 以max_d 和 min_d创建存储元素出现次数的数组
        tmp_arr = [0] * (max_d - min_d + 1)
        # 统计出现的次数
        for num in nums:
            # 数组的起始位置是0，所以需要当前num-min_d寻找下标
            tmp_arr[num - min_d] += 1
        
        # 还原数据
        j = 0
        # 总计需要还原n的数据，故循环n次
        for i in range(n):
            # 剔除tmp_arr中为0的
            while tmp_arr[j] == 0:
                j += 1
            # 当j对应元素值不为0
            # 则将其还原到nums中
            nums[i] = j + min_d
            tmp_arr[j] -= 1
        return nums

 class Solution:
    def sortArray(self, nums: List[int]) -> List[int]:
        # 4.使用计数排序（升序），算法时间复杂度O（n）
        '''
        这个方法对数据要求很高，如果我数据范围太大而且十分稀疏就不太好用了。
        比较适合排数据范围小但是每个数据点有很多个的那种数据
        本题因为数据元素，其速度小于快排
        '''
        # 思想：找到nums中的最大_max和最小数据_min
        # 并以_max和_min为范围大小建立数组tmp_arr
        # tmp_arr的下标是对应nums中的数据值减_min
        # 遍历原始nums数据，并num出现的次数，累加到tmp_arr[num-_min]中去
        # 然后遍历tmp_arr，还原数据

        # 特殊值判断
        if not nums:
            return []

        n = len(nums)

        # 得到数组的最值
        min_d, max_d = min(nums), max(nums)
        # 以max_d 和 min_d创建存储元素出现次数的数组
        tmp_arr = [0] * (max_d - min_d + 1)
        # 统计出现的次数
        for num in nums:
            # 数组的起始位置是0，所以需要当前num-min_d寻找下标
            tmp_arr[num - min_d] += 1
        
        # 还原数据
        i, j = 0, 0
        # 当i< n 时，不断进行数据还原
        while i < n:
            # 寻找当前不为空的位置j
            # 特别注意当第一个位置
            while tmp_arr[j] == 0: 
                # 当tmp_arr[j] == 0，移动到下一个j位置
                j += 1
            # 当其不为空，则按照tmp_arr[j]的值大小，
            # 重复输出j个原始元素
            while tmp_arr[j] != 0:
                # 原始元素值 = 当前tmp_arr的下标 + 最小值
                nums[i] = j + min_d
                # i完成当前赋值以后，后移
                i += 1
                tmp_arr[j] -= 1
            
        return nums

```

## 参考

[十大经典排序](https://www.cnblogs.com/onepixel/p/7674659.html)

[二叉堆](https://labuladong.gitbook.io/algo/shu-ju-jie-gou-xi-lie/er-cha-dui-xiang-jie-shi-xian-you-xian-ji-dui-lie)

## 练习

- [ ] 手写快排、归并、堆排序
