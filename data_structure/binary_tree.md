# 二叉树

## 知识点

### 二叉树遍历

**前序遍历**：**先访问根节点**，再前序遍历左子树，再前序遍历右子树
**中序遍历**：先中序遍历左子树，**再访问根节点**，再中序遍历右子树
**后序遍历**：先后序遍历左子树，再后序遍历右子树，**再访问根节点**

注意点

- 以根访问顺序决定是什么遍历
- 左子树都是优先右子树

#### 递归模板

递归实现二叉树遍历非常简单，不同顺序区别仅在于访问父结点顺序

```Python
def preorder_rec(root):
    if root is None:
        return
    visit(root)
    preorder_rec(root.left)
    preorder_rec(root.right)
    return

def inorder_rec(root):
    if root is None:
        return
    inorder_rec(root.left)
    visit(root)
    inorder_rec(root.right)
    return

def postorder_rec(root):
    if root is None:
        return
    postorder_rec(root.left)
    postorder_rec(root.right)
    visit(root)
    return
```

#### [前序非递归](https://leetcode-cn.com/problems/binary-tree-preorder-traversal/)

本质上是图的DFS的一个特例，因此可以用栈来实现

```Python
class Solution:
    def preorderTraversal(self, root: TreeNode) -> List[int]:
        # 当为空节点时
        if not root:
            return []
        # 对栈初始化，定义结果res
        stack, res = [root], []
        while stack:
            node = stack.pop()
            # 因为栈是后近先出的，先序遍历是：根节点，左，右
            # 所以需要遍历完根节点后，需要先遍历右节点，在遍历左节点
            # 这样出栈的时候，利用栈的后进先出，每次左节点先出，右节点后出
            # 既实现了根， 左， 右
            if node:
                res.append(node.val)
                if node.right:
                    stack.append(node.right)
                if node.left:
                    stack.append(node.left)
        return res
```

#### [中序非递归](https://leetcode-cn.com/problems/binary-tree-inorder-traversal/)

```Python
class Solution:
    def inorderTraversal(self, root: TreeNode) -> List[int]:
        # 使用迭代的方法
        stack, res = [], []
        node = root
        # 思想，每次将左边全走完，然后换到右边一个后，
        # 继续把当前右边的左边都走完，不断迭代
        while stack or node is not None:
            # 不断向左子树走，每走一次就将当前节点保存到栈中
            # 模拟栈递归的调用
            if node is not None:
                stack.append(node) # 当前节点入栈
                node = node.left # 遍历下一个左节点            
            # 当前节点为空时，说明左子树走完了，则从栈中弹出节点，
            # 并保存到结果res中
            # 然后转向右边节点，然后继续城府上面的过程
            # 中序遍历时，判断当前节点为空之后，就可以转去从栈中pop一个节点，
            # 因为这时候，pop的是中间节点不影响。因为是按照 左中右的顺序
            else:
                node = stack.pop() # 当节点为空之后，则从栈中出栈一个节点作为当前节点
                res.append(node.val) # 当前节点值入res
                node = node.right
        return res  
```

#### [后序非递归](https://leetcode-cn.com/problems/binary-tree-postorder-traversal/)

```Python
# 核心就是：根节点必须在右节点弹出之后，再弹出
class Solution:
    def postorderTraversal(self, root: TreeNode) -> List[int]:
        res = [] # 存储后续遍历结果
        stack = [] # 借用栈存储节点遍历
        node = root # 初始为根节点
        while stack or node:
            # 当前节点不为空时，沿着当前节点往下不断遍历至叶子节点
            while node:
                stack.append(node) # 第一次是根节点入栈
                # 判断当前节点的左子树是否存在，若存在，继续向左
                # 若不存在转向右子树
                if node.left is not None:
                    node = node.left
                else:
                    node = node.right
            # 循环结束说明走到了叶子节点，没有左右子树了，
            # 该叶子节点即为当前栈顶元素，应该访问了
            node = stack.pop() # 取出栈顶元素进行访问
            res.append(node.val) # 将栈顶元素值，既当前节点的值添加进res
            # stack[-1]是执行完上面那句取出栈顶元素后的栈顶元素
            # 若栈不为空，且当前节点是栈顶元素的左节点，则直接转去遍历右节点
            # 核心就是：根节点必须在右节点弹出之后，再弹出
            if stack and stack[-1].left == node:
                node = stack[-1].right
            # 若左右子树都没有，则强迫退栈
            else:
                node = None
        return res
```

注意点

- 核心就是：根节点必须在右节点弹出之后，再弹出

DFS 深度搜索-从下向上（分治法）

```Python
class Solution:
    def preorderTraversal(self, root: TreeNode) -> List[int]:
        
        if root is None:
            return []
        
        left_result = self.preorderTraversal(root.left)
        right_result = self.preorderTraversal(root.right)
        
        return [root.val] + left_result + right_result
```

注意点：

> DFS 深度搜索（从上到下） 和分治法区别：前者一般将最终结果通过指针参数传入，后者一般递归返回结果最后合并

#### [BFS 层次遍历](https://leetcode-cn.com/problems/binary-tree-level-order-traversal/)

```Python
from collections import deque
class Solution:
    def levelOrder(self, root: TreeNode) -> List[List[int]]:
        # 若root节点为空，则直接返回
        if not root:
            return []
        # 下面使用队列实现BFS
        
        # 结果集合res
        res = []
        layer = deque()
        # 压入初始节点root
        layer.append(root)
        while layer:
            # 设置临时变量，记录当前层的节点
            cur_layer = []
            # 遍历某一层的节点
            for _ in range(len(layer)):
                # 弹出待处理节点
                node = layer.popleft()
                # 当前节点值如临时cur_layer
                cur_layer.append(node.val)
                # 判断当前node是否有左右节点，
                # 如果有的话，按照先左节点，后右节点的顺序入队列
                if node.left:
                    layer.append(node.left)
                if node.right:
                    layer.append(node.right)
            # 遍历完某层之后，将此层的结果，加入到res中
            res.append(cur_layer)
        # 返回res
        return res
```

### 分治法应用

先分别处理局部，再合并结果

适用场景

- 快速排序
- 归并排序
- 二叉树相关问题

分治法模板

- 递归返回条件
- 分段处理
- 合并结果

常见题目示例

#### [maximum-depth-of-binary-tree](https://leetcode-cn.com/problems/maximum-depth-of-binary-tree/)

> 给定一个二叉树，找出其最大深度。

思路 1：分治法

```Python
# 关键点：此树的深度和其左（右）子树的深度之间的关系。显然，此树的深度等于左子树的深度与右子树的深度中的最大值 +1 。
class Solution:
    def maxDepth(self, root: TreeNode) -> int:
        
        if root is None:
            return 0
        
        return 1 + max(self.maxDepth(root.left), self.maxDepth(root.right))
```

思路 2：层序遍历

```Python
from collections import deque
class Solution:
    def maxDepth(self, root: TreeNode) -> int:
        # 采用层序遍历方法，既BFS
        if not root:
            return 0
        # 定义deque存出每层
        layer = deque()
        layer.append(root)
        depth = 0
        while layer:
            for _ in range(len(layer)):
                node = layer.popleft()
                if node.left:
                    layer.append(node.left)
                if node.right:
                    layer.append(node.right)
            depth += 1
        return depth
```

#### [balanced-binary-tree](https://leetcode-cn.com/problems/balanced-binary-tree/)

> 给定一个二叉树，判断它是否是高度平衡的二叉树。

思路 1：分治法，左边平衡 && 右边平衡 && 左右两边高度 <= 1，

```Python
class Solution:
    def isBalanced(self, root: TreeNode) -> bool:
        # 利用分治法，判断每次节点的递归是否满足
        # 左边平衡&&右边平衡&&左右两边的高度差<=1
        def depth(root):
            # 每当遍历当叶子节点时，返回当前深度为0，且当前为平衡的
            if root is None:
                return 0, True
            # 然后不断遍历左子树和右子树
            depth_l, bool_l = depth(root.left)
            depth_r, bool_r = depth(root.right)

            # 返回左右子树的深度+1（可选），同时判断是否满足
            # 左边平衡&&右边平衡&&左右两边的高度差<=1
            return max(depth_l, depth_r) + 1, bool_l and bool_r and abs(depth_r - depth_l) <= 1
        _, out = depth(root)
        return out
```

思路 2：使用后序遍历实现分治法的迭代版本

```Python
class Solution:
    def isBalanced(self, root: TreeNode) -> bool:

        s = [[TreeNode(), -1, -1]]
        node, last = root, None
        while len(s) > 1 or node is not None:
            if node is not None:
                s.append([node, -1, -1])
                node = node.left
                if node is None:
                    s[-1][1] = 0
            else:
                peek = s[-1][0]
                if peek.right is not None and last != peek.right:
                    node = peek.right
                else:
                    if peek.right is None:
                        s[-1][2] = 0
                    last, dl, dr = s.pop()
                    if abs(dl - dr) > 1:
                        return False
                    d = max(dl, dr) + 1
                    if s[-1][1] == -1:
                        s[-1][1] = d
                    else:
                        s[-1][2] = d
        
        return True
```

#### [binary-tree-maximum-path-sum](https://leetcode-cn.com/problems/binary-tree-maximum-path-sum/)

> 给定一个**非空**二叉树，返回其最大路径和。

思路：分治法。最大路径的可能情况：左子树的最大路径，右子树的最大路径，或通过根结点的最大路径。其中通过根结点的最大路径值等于以左子树根结点为端点的最大路径值加以右子树根结点为端点的最大路径值再加上根结点值，这里还要考虑有负值的情况即负值路径需要丢弃不取。

```Python
class Solution:
    def maxPathSum(self, root: TreeNode) -> int:
        
        self.maxPath = float('-inf')
        
        def largest_path_ends_at(node):
            if node is None:
                return float('-inf')
            
            e_l = largest_path_ends_at(node.left)
            e_r = largest_path_ends_at(node.right)
            
            self.maxPath = max(self.maxPath, node.val + max(0, e_l) + max(0, e_r), e_l, e_r)
            
            return node.val + max(e_l, e_r, 0)
        
        largest_path_ends_at(root)
        return self.maxPath
```

#### [lowest-common-ancestor-of-a-binary-tree](https://leetcode-cn.com/problems/lowest-common-ancestor-of-a-binary-tree/)

> 给定一个二叉树, 找到该树中两个指定节点的最近公共祖先。

思路：分治法，有左子树的公共祖先或者有右子树的公共祖先，就返回子树的祖先，否则返回根节点

```Python
class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        
        if root is None:
            return None
        
        if root == p or root == q:
            return root
        
        left = self.lowestCommonAncestor(root.left, p, q)
        right = self.lowestCommonAncestor(root.right, p, q)
        
        if left is not None and right is not None:
            return root
        elif left is not None:
            return left
        elif right is not None:
            return right
        else:
            return None
```

### BFS 层次应用

#### [binary-tree-zigzag-level-order-traversal](https://leetcode-cn.com/problems/binary-tree-zigzag-level-order-traversal/)

> 给定一个二叉树，返回其节点值的锯齿形层次遍历。Z 字形遍历

思路：在BFS迭代模板上改用双端队列控制输出顺序

```Python
class Solution:
    def zigzagLevelOrder(self, root: TreeNode) -> List[List[int]]:
        
        levels = []
        if root is None:
            return levels
        
        s = collections.deque([root])

        start_from_left = True
        while len(s) > 0:
            levels.append([])
            level_size = len(s)
            
            if start_from_left:
                for _ in range(level_size):
                    node = s.popleft()
                    levels[-1].append(node.val)
                    if node.left is not None:
                        s.append(node.left)
                    if node.right is not None:
                        s.append(node.right)
            else:
                for _ in range(level_size):
                    node = s.pop()
                    levels[-1].append(node.val)
                    if node.right is not None:
                        s.appendleft(node.right)
                    if node.left is not None:
                        s.appendleft(node.left)
            
            start_from_left = not start_from_left
            
        
        return levels
```

### 二叉搜索树应用

####  [validate-binary-search-tree](https://leetcode-cn.com/problems/validate-binary-search-tree/)

> 给定一个二叉树，判断其是否是一个有效的二叉搜索树。

思路 1：中序遍历后检查输出是否有序，缺点是如果不平衡无法提前返回结果， 代码略

思路 2：分治法，一个二叉树为合法的二叉搜索树当且仅当左右子树为合法二叉搜索树且根结点值大于右子树最小值小于左子树最大值。缺点是若不用迭代形式实现则无法提前返回，而迭代实现右比较复杂。

```Python
class Solution:
    def isValidBST(self, root: TreeNode) -> bool:
        
        if root is None: return True
        
        def valid_min_max(node):
            
            isValid = True
            if node.left is not None:
                l_isValid, l_min, l_max = valid_min_max(node.left)
                isValid = isValid and node.val > l_max
            else:
                l_isValid, l_min = True, node.val

            if node.right is not None:
                r_isValid, r_min, r_max = valid_min_max(node.right)
                isValid = isValid and node.val < r_min
            else:
                r_isValid, r_max = True, node.val

                
            return l_isValid and r_isValid and isValid, l_min, r_max
        
        return valid_min_max(root)[0]
```

思路 3：利用二叉搜索树的性质，根结点为左子树的右边界，右子树的左边界，使用先序遍历自顶向下更新左右子树的边界并检查是否合法，迭代版本实现简单且可以提前返回结果。

```Python
class Solution:
    def isValidBST(self, root: TreeNode) -> bool:
        
        if root is None:
            return True
        
        s = [(root, float('-inf'), float('inf'))]
        while len(s) > 0:
            node, low, up = s.pop()
            if node.left is not None:
                if node.left.val <= low or node.left.val >= node.val:
                    return False
                s.append((node.left, low, node.val))
            if node.right is not None:
                if node.right.val <= node.val or node.right.val >= up:
                    return False
                s.append((node.right, node.val, up))
        return True
```

#### [insert-into-a-binary-search-tree](https://leetcode-cn.com/problems/insert-into-a-binary-search-tree/)

> 给定二叉搜索树（BST）的根节点和要插入树中的值，将值插入二叉搜索树。 返回插入后二叉搜索树的根节点。

思路：如果只是为了完成任务则找到最后一个叶子节点满足插入条件即可。但此题深挖可以涉及到如何插入并维持平衡二叉搜索树的问题，并不适合初学者。

```Python
class Solution:
    def insertIntoBST(self, root: TreeNode, val: int) -> TreeNode:
        
        if root is None:
            return TreeNode(val)
        
        node = root
        while True:
            if val > node.val:
                if node.right is None:
                    node.right = TreeNode(val)
                    return root
                else:
                    node = node.right
            else:
                if node.left is None:
                    node.left = TreeNode(val)
                    return root
                else:
                    node = node.left
```

## 总结

- 掌握二叉树递归与非递归遍历
- 理解 DFS 前序遍历与分治法
- 理解 BFS 层次遍历

## 练习

- [ ] [maximum-depth-of-binary-tree](https://leetcode-cn.com/problems/maximum-depth-of-binary-tree/)
- [ ] [balanced-binary-tree](https://leetcode-cn.com/problems/balanced-binary-tree/)
- [ ] [binary-tree-maximum-path-sum](https://leetcode-cn.com/problems/binary-tree-maximum-path-sum/)
- [ ] [lowest-common-ancestor-of-a-binary-tree](https://leetcode-cn.com/problems/lowest-common-ancestor-of-a-binary-tree/)
- [ ] [binary-tree-level-order-traversal](https://leetcode-cn.com/problems/binary-tree-level-order-traversal/)
- [ ] [binary-tree-level-order-traversal-ii](https://leetcode-cn.com/problems/binary-tree-level-order-traversal-ii/)
- [ ] [binary-tree-zigzag-level-order-traversal](https://leetcode-cn.com/problems/binary-tree-zigzag-level-order-traversal/)
- [ ] [validate-binary-search-tree](https://leetcode-cn.com/problems/validate-binary-search-tree/)
- [ ] [insert-into-a-binary-search-tree](https://leetcode-cn.com/problems/insert-into-a-binary-search-tree/)
