# 二分搜索

## 二分搜索模板

给一个**有序数组**和目标值，找第一次/最后一次/任何一次出现的索引，如果没有出现返回-1

模板四点要素

- 1、初始化：start=0、end=len-1
- 2、循环退出条件：start + 1 < end
- 3、比较中点和目标值：A[mid] ==、 <、> target
- 4、判断最后两个元素是否符合：A[start]、A[end] ? target

时间复杂度 O(logn)，使用场景一般是有序数组的查找

典型示例

[binary-search](https://leetcode-cn.com/problems/binary-search/)

> 给定一个  n  个元素有序的（升序）整型数组  nums 和一个目标值  target  ，写一个函数搜索  nums  中的 target，如果目标值存在返回下标，否则返回 -1。

```Python
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        
        l, r = 0, len(nums) - 1
        
        while l + 1 < r:
            mid = (l + r) // 2
            if nums[mid] < target:
                l = mid
            else:
                r = mid

        if nums[l] == target:
            return l
        elif nums[r] == target:
            return r
        else:
            return -1
```

大部分二分查找类的题目都可以用这个模板，然后做一点特殊逻辑即可

另外二分查找还有一些其他模板如下图，大部分场景模板 3 都能解决问题，而且还能找第一次/最后一次出现的位置，应用更加广泛

![binary_search_template](https://img.fuiboom.com/img/binary_search_template.png)

所以用模板#3 就对了，详细的对比可以这边文章介绍：[二分搜索模板](https://leetcode-cn.com/explore/learn/card/binary-search/212/template-analysis/847/)

如果是最简单的二分搜索，不需要找第一个、最后一个位置、或者是没有重复元素，可以使用模板 1，代码更简洁

```Python
# 无重复元素搜索时，更方便
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        
        l, r = 0, len(nums) - 1
        
        while l <= r:
            mid = (l + r) // 2
            if nums[mid] == target:
                return mid
            elif nums[mid] > target:
                r = mid - 1
            else:
                l = mid + 1
        
        return -1
# 如果找不到，start 是第一个大于target的索引
# 如果在B+树结构里面二分搜索，可以return start
# 这样可以继续向子节点搜索，如：node:=node.Children[start]
```

模板 2：

```Python
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        
        l, r = 0, len(nums)
        
        while l < r:
            mid = (l + r) // 2
            if nums[mid] < target:
                l = mid + 1
            else:
                r = mid

        if l < len(nums) and nums[l] == target:
            return l
        
        return -1
```



## 常见题目

### [search-for-range](https://www.lintcode.com/problem/search-for-a-range/description)

> 给定一个包含 n 个整数的排序数组，找出给定目标值 target 的起始和结束位置。
> 如果目标值不在数组中，则返回`[-1, -1]`

思路：核心点就是找第一个 target 的索引，和最后一个 target 的索引，所以用两次二分搜索分别找第一次和最后一次的位置

```Python
class Solution:
    def searchRange(self, nums: List[int], target: int) -> List[int]:
        # 使用两次二分查找进行计算
        # 采用模板三的边界计算：

        # 默认Range范围
        Range = [-1, -1]

        # 先进行容错处理
        if len(nums) == 0:
            return Range
        '''
        0 1 2 3 4 5
        5 7 7 8 8 10
        l   m m'     r
            l'r'
        
        '''
        # 先使用二分查找，找到最左边的范围
        l, r = 0, len(nums)-1
        while(l + 1 < r):
            mid = (l + r) // 2
            # 当中值大于等于target时，不断将右边边界向左移动
            # 因为右边的相等的，此时都不算，不断向左移动右边边界r
            # 此时找到的是target左边的边界
            if nums[mid] >= target:
                r = mid
            else:
                l = mid
        # 尾处理判断最后l和r谁满足结果
        # 一般情况下，当前l刚好满足条件
        if nums[l] == target:
            Range[0] = l
        elif nums[r] == target:
            Range[0] = r
        else:
            # 若都不满足，说明没有找到目标值，直接范围[-1,-1]
            return Range
        
        # 再次使用二分查找，寻找另外一部分
        l, r = 0, len(nums) - 1
        while (l+1 < r):
            mid = (l + r) // 2
            # 当中值小于等于target时，都将左边边界向右移动
            # 因为左边的相等的，此时都不算，不断向右移动左边边界l
            if nums[mid] <= target:
                l = mid
            else:
                r = mid
        # 这个时候，左右应该都相等，但是此时应该以右为先
        # 判断右边边界l应该正好满足
        if nums[r] == target:
            Range[1] = r
        elif nums[l] == target:
            Range[1] = l

        return Range
```

### [search-insert-position](https://leetcode-cn.com/problems/search-insert-position/)

> 给定一个排序数组和一个目标值，在数组中找到目标值，并返回其索引。如果目标值不存在于数组中，返回它将会被按顺序插入的位置。

思路：使用模板 1，若不存在，随后的左边界为第一个大于目标值的索引（插入位置），右边界为最后一个小于目标值的索引

```Python
class Solution:
    def searchInsert(self, nums: List[int], target: int) -> int:
        # 看到排序数组，一般想到二分查找
        # 二分查找先找一下是否存在，若存在返回索引
        # 若不存在，找到合适的插入位置
        # 使用模板1也可以，需要画图自己理解
        if len(nums) == 0:
            return0
        
        l, r = 0, len(nums)-1
        while l <= r:
            mid = (l + r) // 2
            if nums[mid] == target:
                return mid
            elif nums[mid] < target:
                l = mid + 1
            else:
                r = mid - 1
        return l


class Solution:
    def searchInsert(self, nums: List[int], target: int) -> int:
        # 看到排序数组，一般想到二分查找
        # 二分查找先找一下是否存在，若存在返回索引
        # 若不存在，找到合适的插入位置
        # 使用模板3好些
        # 处理数组为空的情况
        if len(nums) == 0:
            return 0
        
        # 需要先判断这个数和当前nums[-1]的大小关系
        # 若其比最后一个还大，说明应该扩容数组
        if target > nums[-1]:
            return len(nums)
        
        # 若目标值比首元素小，则直接插入0位置
        if target < nums[0]:
            return 0
        
        # 若目标值位置在0-(len(nums)-1)之间，则进行二分查找
        l, r = 0, len(nums)-1
        while l + 1 < r:
            mid = (l + r) // 2
            if nums[mid] < target:
                l = mid
            else:
                r = mid
        # 判断和左右哪一个相等
        if nums[l] == target:
            return l
        elif nums[r] == target:
            return r
        # 若和左右两个都不相等，直接插入到r
        else:
            return r
```

### [search-a-2d-matrix](https://leetcode-cn.com/problems/search-a-2d-matrix/)

> 编写一个高效的算法来判断  m x n  矩阵中，是否存在一个目标值。该矩阵具有如下特性：
>
> - 每行中的整数从左到右按升序排列。
> - 每行的第一个整数大于前一行的最后一个整数。



```Python
class Solution:
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        # 先以右上角为起始判断点，初步判断target在哪一行
        # 定位到行之后，对进行二分查找
        # 异常筛选,若二维数组为空
        if len(matrix) ==0 or len(matrix[0]) ==0:
            return False

        h, w = len(matrix), len(matrix[0])

        # 使用循环遍历行，对最后一列进行判断
        for i in range(h):
            # 若target大于当前行的最大值（既最后一列的值）
            # 则到下一行去
            if matrix[i][w-1] < target:
                continue
            
            # 若target在此行，则进行二分查找
            l , r= 0, w-1
            while l + 1 < r:
                mid = (l + r) // 2
                if matrix[i][mid] < target:
                    l = mid
                else:
                    r = mid
                
            # 尾处理
            if matrix[i][l] == target or matrix[i][r] == target:
                return True
            else:
                return False
        # 若都遍历完了，没有zhdao
        return False

思路：两次二分，首先定位行数，接着定位列数       
class Solution:
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        # 使用两次二分查找，
        # 第一次在每行第一列进行查找，确定target所在行
        # 第二次在定位行中进行查找
        if len(matrix) ==0 or len(matrix[0]) ==0:
            return False
        '''
        使用模板1的几点总结说明：
        1.若寻找到最后，没有找到目标值：
        1）若目标值小于最后的mid值，最终位置关系为：R < target < L(Mid)
        2）若目标值大于最后的mid值，最终位置关系为：            R(Mid) < target < L
        '''
        # 当没有找到时，以最后的R为起始行，
        l, r = 0, len(matrix) - 1

        while l <= r:
            mid = (l + r) // 2
            if matrix[mid][0] == target:
                return True
            elif matrix[mid][0] < target:
                l = mid + 1
            else:
                r = mid - 1
        
        # 若没有找到进行二次行查找
        row = r
        l, r = 0, len(matrix[0]) - 1
        while l <= r:
            mid = (l + r) // 2
            if matrix[row][mid] == target:
                return True
            elif matrix[row][mid] < target:
                l = mid + 1
            else:
                r = mid - 1
        
        # 若循环执行完了，说明没有
        return False
        


class Solution:
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        # 利用此二维数组的特点，其可以视为是一个一维有序数组
        # 直接利用二分查找进行
        # 使用坐标时，需要进行转换，一维位置转换成二维

        # 异常筛选,若二维数组为空
        if len(matrix) ==0 or len(matrix[0]) ==0:
            return False

        h, w = len(matrix), len(matrix[0])

        l , r= 0, w * h -1

        while l + 1 < r:
            mid = (l + r) // 2
            # 一维变二维位置坐标：
            cur_h = mid // w
            cur_w = mid % w

            if matrix[cur_h][cur_w] < target:
                l = mid
            else:
                r = mid
        # 现将当前l r转换为 二维坐标
        l_h =  l // w
        l_w = l % w

        r_h =  r // w
        r_w = r % w

        # 尾处理
        if matrix[l_h][l_w] == target or matrix[r_h][r_w] == target:
            return True
        else:
            return False     
```

### [first-bad-version](https://leetcode-cn.com/problems/first-bad-version/)

> 假设你有 n 个版本 [1, 2, ..., n]，你想找出导致之后所有版本出错的第一个错误的版本。
> 你可以通过调用  bool isBadVersion(version)  接口来判断版本号 version 是否在单元测试中出错。实现一个函数来查找第一个错误的版本。你应该尽量减少对调用 API 的次数。

```Python
class Solution:
    def firstBadVersion(self, n):
        
        l, r = 1, n
        
        while l + 1 < r:
            mid = (l + r) // 2
            if isBadVersion(mid):
                r = mid
            else:
                l = mid
        
        if isBadVersion(l):
            return l
        else:
            return r
```
```python
class Solution:
    def firstBadVersion(self, n):
        """
        :type n: int
        :rtype: int
        """
        # 使用二分法进行查找
        # 使用模板3最后都会留下两个值
        # 最两边的边界值，最后也可以的遍历到
        # 不用纠结
        # 最特殊情况全F，最终LR在倒数第一，第二位置
        # 分别判断，先判断左边是否为T，剩下的最边界的出错地方
        # 还有全T情况，最终LR在第一，第二位置
        # 若第一开始错，则就是1，反之就是第二开始错

        # 题目默认了，出错的位置，至少出现了在1 和 n上
        l, r = 1, n

        while l + 1 < r:
            mid = (l + r) // 2
            if isBadVersion(mid) == False:
                l = mid
            else:
                r = mid
        if isBadVersion(l):
            return l
        else:
            return r
```           

### [find-minimum-in-rotated-sorted-array](https://leetcode-cn.com/problems/find-minimum-in-rotated-sorted-array/)

> 假设按照升序排序的数组在预先未知的某个点上进行了旋转( 例如，数组  [0,1,2,4,5,6,7] 可能变为  [4,5,6,7,0,1,2] )。
> 请找出其中最小的元素。假设数组中无重复元素。

思路：使用二分搜索，当中间元素大于右侧元素时意味着拐点即最小元素在右侧，否则在左侧

```Python
class Solution:
    def findMin(self, nums: List[int]) -> int:
        
        l , r = 0, len(nums) - 1
        
        while l < r:
            mid = (l + r) // 2
            if nums[mid] > nums[r]: # 数组有重复时，若 nums[l] == nums[mid] == nums[r]，无法判断移动方向
                l = mid + 1
            else:
                r = mid
        
        return nums[l]
```
![binarySearch-findMin](https://github.com/BonesCat/algorithm-pattern-python/tree/master/images/binarySearch-findMin.jpg)

```Python
class Solution:
    def findMin(self, nums: List[int]) -> int:
        l, r = 0, len(nums) - 1
        # 本题目的数组中不存在相等的数
        # 所提l, mid, r之间只存在大小不等关系
        while l + 1 < r:
            mid = (l + r) // 2
            # 如果中值 < 右值，则最小值在左半边，可以收缩右边界。
            # 如果中值 > 右值，则最小值在右半边，可以收缩左边界。
            # 中值 > 右值，最小值在右半边，收缩左边界
            # 因为中值 > 右值，中值肯定不是最小值，左边界可以跨过mid

            if nums[mid] > nums[r]:
                l = mid
            # 若mid小于right，说明右侧为递增
            # 说明最小应该出现在左侧，所以
            # 将mid设置为右边界
            else:
                r = mid
        # 循环结束，剩下两个值，进行尾处理
        # 判断哪个值更小，即为所需要的
        return min(nums[l], nums[r])
```

### [find-minimum-in-rotated-sorted-array-ii](https://leetcode-cn.com/problems/find-minimum-in-rotated-sorted-array-ii/)

> 假设按照升序排序的数组在预先未知的某个点上进行了旋转
> ( 例如，数组  [0,1,2,4,5,6,7] 可能变为  [4,5,6,7,0,1,2] )。
> 请找出其中最小的元素。(包含重复元素)
这个和上个题目看这个大神的题解：

https://leetcode-cn.com/problems/find-minimum-in-rotated-sorted-array/solution/tong-guo-hua-tu-geng-neng-shen-ke-li-jie-er-fen-fa/

https://leetcode-cn.com/problems/find-minimum-in-rotated-sorted-array-ii/solution/154-find-minimum-in-rotated-sorted-array-ii-by-jyd/

https://leetcode-cn.com/problems/find-minimum-in-rotated-sorted-array-ii/solution/tong-guo-hua-tu-lai-shen-ke-li-jie-er-fen-fa-by-ch/

```Python
class Solution:
    def findMin(self, nums: List[int]) -> int:
        l, r = 0, len(nums) - 1
        # 本题目的数组中不存在相等的数
        # 所提l, mid, r之间只存在大小不等关系
        while l + 1 < r:
            mid = (l + r) // 2
            # 如果中值 < 右值，则最小值在左半边，可以收缩右边界。
            # 如果中值 > 右值，则最小值在右半边，可以收缩左边界。
            # 中值 > 右值，最小值在右半边，收缩左边界
            # 因为中值 > 右值，中值肯定不是最小值，左边界可以跨过mid

            if nums[mid] > nums[r]:
                l = mid
            # 若mid小于right，说明右侧为递增
            # 说明最小应该出现在左侧，所以
            # 将mid设置为右边界
            else:
                r = mid
        # 循环结束，剩下两个值，进行尾处理
        # 判断哪个值更小，即为所需要的
        return min(nums[l], nums[r])      
```



### [search-in-rotated-sorted-array](https://leetcode-cn.com/problems/search-in-rotated-sorted-array/)

> 假设按照升序排序的数组在预先未知的某个点上进行了旋转。
> ( 例如，数组  [0,1,2,4,5,6,7]  可能变为  [4,5,6,7,0,1,2] )。
> 搜索一个给定的目标值，如果数组中存在这个目标值，则返回它的索引，否则返回  -1 。
> 你可以假设数组中不存在重复的元素。

![binarySearch-searchTargetInRotateArray](https://github.com/BonesCat/algorithm-pattern-python/tree/master/images/binarySearch-searchTargetInRotateArray.jpg)
```Python
class Solution:
    def search(self, nums, target):
        # 方法，先使用二分法确定分界点
        # 此数组中不包含重复元素
        # 在根据target和分界点值的关系确定target所述区域
        # 再次使用一次二分法

        # 先进行特殊判断
        if len(nums) == 0:
            return -1
        # 模板3：
        l, r = 0, len(nums) - 1

        while l + 1 < r:
            mid = (l + r) // 2
            if nums[mid] > nums[r]:
                l = mid
            else:
                r = mid
        # 尾处理，确定分界位置
        # 找最小的那个的位置
        if nums[l] < nums[r]:
            depart = l
        else:
            depart = r
        
        # 通过target与depart关系确定二次搜索区域
        if target == nums[depart]:
            return depart
        elif depart == 0:
            # 此时是升序，则直接进行搜索
            l, r = 0, len(nums)-1
        elif nums[0] <= target <= nums[depart-1]:
            l, r = 0, depart-1
        else:
            l, r = depart, len(nums)-1
        
        while l + 1 < r:
            mid = (l + r) // 2
            if nums[mid] > target:
                r = mid
            else:
                l = mid
        # 尾处理，确定最后是否存在
        if nums[l] == target:
            return l
        elif nums[r] == target:
            return r
        else:
            return -1
            
class Solution2:
    def search(self, nums, target):
        # 方法，使用二分法一次性进行判断
        # 此数组中不包含重复元素


        # 先进行特殊判断
        if len(nums) == 0:
            return -1
        # 模板3：
        l, r = 0, len(nums) - 1
        while l + 1 < r:
            mid = (l + r) // 2
            # 先判断target和end的关系，确定其大致在什么部分
            # 在较大的升序部分（既折断，翻转的部分）
            if target > nums[r]:
                # 若mid在target左边，则缩小左边界到mid
                if nums[mid] > target or nums[mid] < nums[r]:
                    r = mid
                # 否则缩小右边界
                else:
                    r = mid
            else:
                if nums[mid] < target or nums[mid] > nums[r]:
                    l = mid
                else:
                    r = mid
        print(l, r)

        # 尾处理，确定最后是否存在
        if nums[l] == target:
            return l
        elif nums[r] == target:
            return r
        else:
            return -1
```

注意点

> 面试时，可以直接画图进行辅助说明，空讲很容易让大家都比较蒙圈

### [search-in-rotated-sorted-array-ii](https://leetcode-cn.com/problems/search-in-rotated-sorted-array-ii/)

> 假设按照升序排序的数组在预先未知的某个点上进行了旋转。
> ( 例如，数组  [0,0,1,2,2,5,6]  可能变为  [2,5,6,0,0,1,2] )。
> 编写一个函数来判断给定的目标值是否存在于数组中。若存在返回  true，否则返回  false。(包含重复元素)

```Python
class Solution:
    def search(self, nums: List[int], target: int) -> bool:
        # 方法，使用二分法一次性进行判断
        # 此数组中不包含重复元素
        # 先进行特殊判断
        if len(nums) == 0:
            return False
        # 模板3：
        l, r = 0, len(nums) - 1
        while l + 1 < r:
            mid = (l + r) // 2
            # 判断中值是否等于target
            if nums[mid] == target:
                return True
            # 先判断中值和end的关系，确定其mid在左还是右部分           
            # 如果nums[mid]>nums[right],说明mid在左边部分（既折断，翻转的部分）
            if nums[mid] > nums[r]:
                # 此时左半段是有序的
                # 在此左半端判断target是否在其中
                # 若在，则缩小右边界至mid
                if nums[l] <= target < nums[mid]:
                    r = mid
                # 否则缩小左边界至mid
                else:
                    l = mid
            # 否则的话，说明mid在右边部分
            elif nums[mid] < nums[r]:
                # 此时，右边这部分有序，可以判断target和mid和right的大小关小
                # 若nums[mid] < target < nums[r]
                # 说明target在此部分，缩小左边界
                # 都直接将l移动至mid
                if nums[mid] < target <= nums[r]:
                    l = mid
                # 反之，target在剩下的左边+一点mid的位置
                # 缩小右边边界
                else:
                    r = mid
            else:
                # 若此时nums[mid]和nums[r]相等
                # 相同元素时，将右边节向中间压缩
                # 将r向左移动一个
                r -= 1

        # 尾处理，确定最后是否存在
        if nums[l] == target or nums[r] == target:
            return True
        else:
            return False
```

## 总结

二分搜索核心四点要素（必背&理解）

- 1、初始化：start=0、end=len-1
- 2、循环退出条件：start + 1 < end
- 3、比较中点和目标值：A[mid] ==、 <、> target
- 4、判断最后两个元素是否符合：A[start]、A[end] ? target

## 练习题

- [ ] [search-for-range](https://www.lintcode.com/problem/search-for-a-range/description)
- [ ] [search-insert-position](https://leetcode-cn.com/problems/search-insert-position/)
- [ ] [search-a-2d-matrix](https://leetcode-cn.com/problems/search-a-2d-matrix/)
- [ ] [first-bad-version](https://leetcode-cn.com/problems/first-bad-version/)
- [ ] [find-minimum-in-rotated-sorted-array](https://leetcode-cn.com/problems/find-minimum-in-rotated-sorted-array/)
- [ ] [find-minimum-in-rotated-sorted-array-ii](https://leetcode-cn.com/problems/find-minimum-in-rotated-sorted-array-ii/)
- [ ] [search-in-rotated-sorted-array](https://leetcode-cn.com/problems/search-in-rotated-sorted-array/)
- [ ] [search-in-rotated-sorted-array-ii](https://leetcode-cn.com/problems/search-in-rotated-sorted-array-ii/)
