

Python 多元運算思維設計
===

> Python 多元運算思維設計- 程序式、物件式與函數式程設
> *Python Multimodal Thinking Design - Procedural, Object-Oriented, and Functional Programming*

程式設計有很多種思維的作法，例如程序(procedure-based)的方式- 程式是由許多的程序所構成; 例如物件導向(object-oriented)的方式- 其設計的主體是物件; 又例如本章所要介紹的，程式的主體是一個個的函數(function programming)。

# 函數式程設

函數式程設（Functional Programming，簡稱 FP）是一種程設範式(programming paradigm)，它主要基於數學中的函數概念和函數的應用。在函數式程設中，計算被視為一系列函數之間的數學變換，而不是一系列狀態的改變。以下是函數式程設的一些主要特點：


## 函數是主角

* 高階函式 (high order function)
* 匿名函示 (anonymous function)
* 純函式 (pure function)

## FP  的特點

* 不可變性（Immutability）：函數式程設強調不修改資料，而是建立新的資料。這有助於避免意外的狀態變化和提高程式碼的可靠性。
* 純函數（Pure Functions）：純函數是指不依賴外部狀態或產生副作用的函數。相同的輸入總是產生相同的輸出，且不會修改任何全局變數。
* 高階函數（Higher-Order Functions）：函數可以作為另一個函數的參數傳遞，也可以作為函數的回傳值。這種特性使得程式碼更具彈性，可以更容易地抽象和組合功能。
* 遞歸（Recursion）：函數式程設通常使用遞歸代替循環，這樣可以更自然地處理資料結構。
* 不可見的狀態（No Side Effects）：函數式程設努力減少或消除副作用，這意味著函數在執行時不應該修改外部狀態或引起不可預測的行為。
* 基於表達式（Expression-Based）：在函數式程設中，程式碼往往更多地由表達式組成，而不是語句。

函數式程設語言如 Haskell、Clojure、Scala 和 Erlang 以及支持函數式程設風格的特性的語言如 Python、JavaScript 和 Java 都可以用於實現函數式程設的概念。函數式程設的目標是寫出更具可讀性、可維護性和可測試性的程式碼，並且在某些情況下，它可以更好地處理並行性和併發性問題。

當然，我可以使用 Python 實例來說明函數式程設的概念。以下是一些Python程式碼範例，用於說明函數式程設的不同特點：


## 不可變性（Immutability）：

不可變性（Immutability）是函數式程設的一個重要概念，它強調資料一旦被建立就不能被修改。以下是更詳細的例子以及反例，來凸顯不可變性的優點：

**例子1：使用不可變性的字串**

先來看以下的例子：
```python
# 不可變的字串範例
name = "Alice"
new_name = name.upper()  # 建立一個新的字串，而不是修改原始字串
print(name)              # 仍然是 "Alice"
print(new_name)          # "ALICE"
```

`name.upper()` 並不會改變 name 本身的值，而是回傳一個其值的大寫。事實上 str 字串是一個不可變的物件，其所提供的任何方法都不會改變其值，只會回傳一個新的值。如此確保字串的不可變性，提升程式的可維護性。

**反例1：使用可變性的列表**

```python
my_list = [1, 2, 3, 4]
my_list.append(5)  # 直接修改了列表，添加了一個新元素
print(my_list)  # [1, 2, 3, 4, 5]
```
與之相法的，list (列表) 是一個可變物件。在這個反例中，我們使用可變的列表，直接在原始列表上添加了一個新元素。`my_list` 在呼叫 `append()` 之後，其狀態由原有的 `[1,2,3,4]` 變成 `[1,2,3,4,5]`。

如果我們要保留原有列表的狀態，可以在呼叫之前先做一個複製(`.copy()`)。如下：

```
my_list = [1, 2, 3, 4]
same_list = my_list.copy()
my_list.append(5)
print (same_list)
```

注意：一些人可能會用 `same_list = my_list` 但這樣兩個變數都會指到相同的空間，導致 `same_list` 也跟著改變了。

**例子2：不可變性的元組**

```python
point = (3, 4)  # tuple (元組)
print(point[0])  # 3
# point[0] = 5  # 此行程式碼將引發錯誤，元組不可修改
```

這個例子展示了元組的不可變性。一旦建立了元組，就無法修改其中的元素。這有助於確保資料的穩定性，特別是在多線程環境中。如果你只想讀取一群資料，而非修改它，利用元組來設計，而非列表是不錯的方式。

**反例2：使用可變性的字典**

如同 list, dict 也是可變的：

```python
alice = {"name": "Alice", "age": 30}
alice["age"] = 31   # 直接修改字典中的值
print(alice)        # {'name': 'Alice', 'age': 31}
```

在這個反例中，我們使用可變的字典，直接修改了字典中的值。這種操作可能導致難以追蹤的狀態變化，並且在程式碼中引入了不必要的複雜性。我們可以透過 .items 生成一個 tuple 物件：

```
alice = {"name": "Alice", "age": 30}
print (alice.items()) 
# dict_items([('name', 'Alice'), ('age', 30)])
```

### BMI example

在 BMI 的例子中，我們將ＸＸ設計成一個不可變的物件

## 純函數（Pure Functions）：

在函數式編程中，純函數（Pure Function）具備以下兩個主要特徵：

1. **相同的輸入總是產生相同的輸出：** 給定相同的輸入，純函數總是返回相同的輸出，而不受程式的狀態或外部因素的影響。這種特性被稱為「引用透明性」(referential transparency)，它使得函數調用的結果可以完全預測，有助於程式的可靠性和測試。

2. **無副作用：** 純函數不應該引起程式狀態的變化，也就是說，在執行時它不應修改任何全域變數、修改輸入參數、印出訊息或執行與計算結果無關的任何可觀察操作。這樣可以避免意外的影響和產生不可預測的行為。

純函數的這兩個特性使其更易於理解、測試和推理。由於純函數不依賴於外部狀態，它們對於並行化和並發性也有天然的支持，因為它們不會導致競爭條件或其他共享狀態問題。

以下是一個簡單的純函數示例：

```python
def add(a, b):
    return a + b
```

這個函數滿足純函數的要求。給定相同的輸入 `a` 和 `b`，它總是返回相同的輸出，而且沒有任何副作用。相反，以下是一個不是純函數的例子：

```python
total = 0

def add_with_side_effect(a):
    global total
    total += a
    return total
```

這個函數有副作用，因為它修改了全域變數 `total`，而且對於相同的輸入 `a`，其輸出也依賴於之前的狀態。

### 非純函數範例（有副作用）
```
def add_and_print(a, b):
    result = a + b
    print(result)  # 印出結果，有副作用
    return result

result = add_and_print(3, 5)  # 回傳結果並且印出
print(result)  # 8
```

### Concurrent 

讓我們來看一個例子，說明一個不符合純函數特性的情況，導致並行化和並發性方面的問題。

考慮以下的 Python 函數：

```python
total = 0

def add_with_side_effect(a):
    global total
    total += a
    return total
```

這個函數具有副作用，因為它修改了全局變數 `total`。現在，想像有兩個並行的任務（tasks）想要同時執行 `add_with_side_effect` 函數，每個任務都希望根據當前的 `total` 值來進行計算。

```python
from concurrent.futures import ThreadPoolExecutor

def concurrent_task(task_id, value):
    result = add_with_side_effect(value)
    print(f"Task {task_id}: Result = {result}")

# 利用 ThreadPoolExecutor 並行執行兩個任務
with ThreadPoolExecutor(max_workers=2) as executor:
    executor.submit(concurrent_task, 1, 5)
    executor.submit(concurrent_task, 2, 3)
```

由於 `add_with_side_effect` 函數具有全局狀態的副作用，這兩個任務可能會交錯地修改 `total`，導致計算結果與期望不符。

這個例子中，由於全局變數的共享和修改，並行執行這些任務可能會導致競爭條件（Race Condition）或其他並行性相關的問題。這種情況下，並行化的優勢被破壞，因為函數的結果取決於全局狀態的修改，並可能導致不確定的行為。

在函数式编程中，鼓勵使用不可變函數，以更好地支持並行化，


## 高階函數（Higher-Order Functions）：

> 在函數式程式設計中，高階函數是指將一個或多個函數作為參數和/或傳回一個函數作為其結果的函數。 本質上，它將函數視為一等公民，允許它們像任何其他資料類型一樣傳遞和操作。

Python 支援高階函數，它們在函數式程式設計範式中發揮重要作用。

高階函數在函數式編程中有許多好處，它們可以提高代碼的靈活性、可讀性和可維護性。以下是一些高階函數的好處：

1. **抽象和模塊化：** 高階函數允許你將功能抽象為函數，這樣可以更好地模塊化代碼。通過將功能封裝在函數中，你可以提高代碼的可讀性和可理解性。
2. **代碼覆用：** 由於高階函數的靈活性，你可以將通用功能定義為函數，並在需要時重覆使用。這有助於減少代碼的冗余，提高了代碼的可維護性。
3. **函數作為一等公民：** 在函數式編程中，函數被視為一等公民，就像其他數據類型一樣。這意味著你可以將函數傳遞給其他函數，將函數作為參數傳遞給函數，或者將函數作為返回值。
4. **適應不同的場景：** 高階函數使得代碼更具適應性。通過將函數作為參數傳遞，你可以輕松更改和適應代碼的行為，而無需修改函數的定義。
5. **簡潔性：** 使用高階函數可以使代碼更加簡潔。通過使用匿名函數（lambda 表達式）和內置的高階函數，可以用更少的代碼表達相同的邏輯。
6. **函數式編程思想：** 高階函數是函數式編程的核心概念之一。使用高階函數可以更好地采用函數式編程的思想，如不可變性和純函數，從而產生更健壯和可維護的代碼。
7. **更強大的組合：** 通過將多個小函數組合成一個覆雜的函數，可以創建出強大且可重用的功能塊。這種組合性質使得代碼更易於理解和維護。

總體而言，高階函數提供了一種更抽象、更靈活的編程方式，使得代碼更易於理解、維護，並能更好地應對不同的需求。這些好處使得高階函數在函數式編程風格中得到了廣泛的應用。

```python
# 高階函數範例
def apply(op, x, y):
    return op(x, y)

def add(a, b):
    return a + b

def subtract(a, b):
    return a - b

d1, d2 = 3, 5
ops = [add, subtract]

for op in ops:
    r = apply(op, d1, d2)
    print (r)
```

### lambda

`lambda` 是Python中用於創建匿名函數的關鍵字。匿名函數是指沒有函數名的函數，通常用於一次性的、簡單的操作。與普通函數不同，`lambda` 函數可以在一行內定義，並且可以作為參數傳遞給高階函數。

以下是關於 `lambda` 函數的一些重要點：

**`lambda` 函數的語法：**

```python
lambda arguments: expression
```

- `lambda` 關鍵字用於聲明匿名函數。
- `arguments` 是函數的參數，類似於普通函數。
- `expression` 是函數體，描述了函數的返回值。

**`lambda` 函數的應用：**

`lambda` 函數通常用於一次性的、簡單的操作，特別是在需要傳遞函數作為參數的情況下，比如高階函數。

**`lambda` 函數作為高階函數的參數：**

由於 `lambda` 函數的簡潔性，它經常用作高階函數的參數，尤其是在需要傳遞簡短功能的情況下。

**使用 `map` 和 `lambda` 函數**

```python
numbers = [1, 2, 3, 4, 5]
squared_numbers = list(map(lambda x: x * x, numbers))
```

在這個示例中，`lambda x: x * x` 是一個匿名函數，它將輸入參數 `x` 的平方作為返回值。該函數被傳遞給 `map` 函數，用於對 `numbers` 列表中的每個元素進行平方操作。

**`lambda` 函數的特性：**

- `lambda` 函數是匿名的，因此沒有函數名。
- `lambda` 函數可以有多個參數，但只能有一個表達式。
- `lambda` 函數可以出現在任何需要函數對象的地方。

**`lambda` 函數與 `functools.partial` 的結合：**

`lambda` 函數常與 `functools.partial` 結合使用，通過部分應用（partial application）函數的參數，創建一個新的函數。

**示例：使用 `functools.partial` 和 `lambda` 函數**

```python
from functools import partial

# 創建部分應用的函數
multiply_by_two = lambda x, y: x * y
partial_function = partial(multiply_by_two, 2)

# 使用部分應用的函數
result = partial_function(5)  # 結果為 10
```

在這個示例中，`partial` 函數用於創建一個新的函數 `partial_function`，該函數是 `multiply_by_two` 函數的部分應用，其中 `y` 參數被固定為 2。

總體而言，`lambda` 函數是一種輕量級的函數定義方式，它在高階函數中特別有用，因為它允許你在需要函數作為參數的地方以更簡潔的方式定義功能。


### `map` 函數

將一個函數應用於可叠代對象的每個元素，返回一個新的可叠代對象。

範例
```python
numbers = [1, 2, 3, 4, 5]
squared_numbers = list(map(lambda x: x * x, numbers))
```

### `filter` 函數

使用一個函數過濾可叠代對象的元素，返回滿足條件的元素組成的新可叠代對象。

範例
```python
numbers = [1, 2, 3, 4, 5]
even_numbers = list(filter(lambda x: x % 2 == 0, numbers))
```

### BMI 的例子
透過 filter 找出 bmi 異常的人

### `reduce` 函數

使用一個函數依次合並可叠代對象的元素，返回一個單一的值。

範例
```python
from functools import reduce
numbers = [1, 2, 3, 4, 5]
product = reduce(lambda x, y: x * y, numbers)
```

### `sorted` 函數

對可叠代對象的元素進行排序，可以接受一個比較函數。

範例
```python
words = ["apple", "banana", "kiwi", "orange", "grape"]
sorted_words = sorted(words, key=lambda x: len(x))
```

這些函數是函數式編程中常見的高階函數，它們提供了一種靈活而強大的方式來處理數據。在上述示例中，`lambda` 表達式用於定義匿名函數，但實際上，你也可以使用命名函數。

### BMI 的例子

將一個群組的人，依據他們的體脂肪來進行排序

## 遞歸（Recursion）：

```python
# 遞歸範例：計算階乘
def factorial(n):
    if n == 0:
        return 1
    else:
        return n * factorial(n - 1)

result = factorial(5)
print(result)  # 120
```

## 不可見的狀態（No Side Effects）：

```python
# 避免副作用範例
def multiply_list_by_constant(lst, constant):
    new_lst = []  # 建立新的列表
    for item in lst:
        new_lst.append(item * constant)  # 不修改原始列表
    return new_lst

my_list = [1, 2, 3, 4]
new_list = multiply_list_by_constant(my_list, 2)
print(my_list)  # [1, 2, 3, 4]（原列表未改變）
print(new_list)  # [2, 4, 6, 8]
```

這些範例展示了如何在Python中應用函數式程設的不同概念，包括不可變性、純函數、高階函數、遞歸和避免副作用。這些概念有助於寫出更具可讀性和可維護性的程式碼，並有助於減少錯誤。

## 一等公民 (first class function)


```python
from typing import Callable

def double(n: int) -> int:
    return 2 * n

def is_even(n: int) -> bool:
    return n % 2 == 0

def make_adder(n: int) -> Callable[int, int]:

    def adder(m: int) -> int:
        return n + m

    return adder

if __name__ == "__main__":
    # 1. Functions can be assigned to variables
    times2 = double
    print(times2(21)) # 42

    # 2. Functions can be passed to other functions arguments
    evens_under10 = list(filter(is_even, range(10)))
    print(evens_under10) # [0, 2, 4, 6, 8]

    # 3. Functions can be returned from functions
    add10 = make_adder(10)
    print(add10(5)) # 15
```

## FP reference

[Functional Programming in Python
Essential concepts, patterns, and modules (Medium)](https://medium.com/akava/functional-programming-in-python-e492f2ad1e37)

## 物件導向程設與函數式程設

當函數式程設（FP）和物件導向程設（OOP）互補使用時，它們可以在不同層面的應用中發揮作用。以下是一個具體的範例，展示了如何在同一個應用中結合這兩種程設範式：

考慮一個圖書管理系統，其中有書籍（Book）和讀者（Reader）這兩個主要的概念。我們可以使用OOP來建模書籍和讀者，因為它們具有屬性和方法。然後，我們可以使用FP來執行某些資料轉換和查詢操作。

### BMI example

* Currency is an immutable class
* when we analyzing the workout record, the record should not be modified 
* class `Workout` encapsulates function about workout, each one is static, so it is a pure function?

### Book example

1. **使用物件導向程設（OOP）建模書籍和讀者**：

```python
class Book:
    def __init__(self, title, author):
        self.title = title
        self.author = author
        self.checked_out = False

    def check_out(self):
        self.checked_out = True

    def check_in(self):
        self.checked_out = False

class Reader:
    def __init__(self, name):
        self.name = name
        self.checked_out_books = []

    def check_out_book(self, book):
        if not book.checked_out:
            self.checked_out_books.append(book)
            book.check_out()
```

在這個範例中，我們建立了兩個類：`Book` 和 `Reader`，它們具有不同的屬性和方法來模擬書籍和讀者的行為。

2. **使用函數式程設（FP）執行資料轉換**：

現在，假設我們想要查找哪些書籍已經被借閱。我們可以使用FP的 `filter` 函數來執行這個操作：

```python
def find_checked_out_books(books):
    return filter(lambda book: book.checked_out, books)

# 建立一些書籍和讀者
book1 = Book("Book 1", "Author A")
book2 = Book("Book 2", "Author B")
reader1 = Reader("Reader X")

# 讓讀者借閱一些書
reader1.check_out_book(book1)
reader1.check_out_book(book2)

# 查找已借閱的書籍
checked_out_books = find_checked_out_books([book1, book2])
for book in checked_out_books:
    print(f"{book.title} by {book.author} is checked out.")
```

在這個範例中，我們使用 `filter` 函數找到了已借閱的書籍，並印出它們的資訊。

這個範例展示了如何在OOP模型中建模實體（書籍和讀者），同時使用FP來執行資料查詢和轉換操作。這兩種程設範式可以互相補充，以實現更好的程式碼組織和可維護性。

### Lambda vs. FP

函數式程設（FP）和 lambda 表達式之間有密切的關係。Lambda 表達式是一種用於建立匿名函數的機制，而匿名函數在函數式程設中非常常見。

以下是關於 FP 和 lambda 表達式之間的關係：

1. **匿名函數**：Lambda 表達式允許你建立匿名函數，這是不具名的、一次性的函數。這些函數通常用於簡單的操作，而不需要額外地定義一個具名函數。

2. **高階函數**：FP 中的高階函數（higher-order functions）是可以接受函數作為參數或回傳函數作為結果的函數。Lambda 表達式通常用於傳遞函數作為參數，使得高階函數更具靈活性。

3. **簡潔性**：Lambda 表達式可以讓你在不需要定義額外函數的情況下，將一個簡單的操作嵌入到程式碼中，使程式碼更緊湊且可讀。

以下是一個使用 lambda 表達式的簡單範例，將其傳遞給 `map` 函數：

```python
numbers = [1, 2, 3, 4, 5]
squared_numbers = map(lambda x: x**2, numbers)
result = list(squared_numbers)
# result 為 [1, 4, 9, 16, 25]
```

在這個範例中，我們使用 lambda 表達式定義了一個匿名函數，該函數接受一個參數 x 並回傳 x 的平方。然後，我們將這個匿名函數傳遞給 `map` 函數，以對列表中的每個元素進行平方操作。

總之，lambda 表達式是函數式程設中的一個有用工具，用於建立匿名函數，並可以與高階函數一起使用，以實現更具彈性的程式碼。

### 集合推導 vs. FP

函數式程設（FP）和集合推導（Set Comprehensions）之間有關聯，因為它們都關注資料轉換和過濾的功能，並強調使用表達式來建立新集合。

集合推導是一種在某些程設語言中常見的語法，用於建立新集合（通常是列表、集合或字典），通過對現有集合進行轉換和過濾操作，以一種簡潔和可讀的方式。這種語法通常包含了類似於 FP 的思想和風格。

以下是集合推導和FP之間的一些相似之處：

1. **轉換和過濾**：集合推導和FP都關注資料的轉換和過濾。它們允許你根據特定的條件對資料集進行操作，並生成新的資料集。

2. **表達式**：在集合推導中，你可以使用表達式來定義如何從現有集合生成新集合。這些表達式可以包括函數式的操作，如映射和過濾。

3. **不可變性**：FP強調不可變性，而集合推導通常生成新的集合，而不是修改原始集合，這符合不可變性的原則。

以下是一個使用集合推導的範例，用於生成一個包含偶數的新集合：

```python
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
even_numbers = {x for x in numbers if x % 2 == 0}
# even_numbers 為 {2, 4, 6, 8, 10}
```

在這個範例中，我們使用了集合推導，通過過濾原始列表中的元素來建立一個新的集合，其中包含所有偶數。這個操作類似於 FP 中的過濾操作。

總之，集合推導和函數式程設都關注資料轉換和操作，它們在設計風格和思維方式上有一些相似之處。集合推導通常用於生成新的集合，而FP則更加廣泛，包括了函數式操作的其他方面。然而，集合推導可以被視為一種函數式程設的特殊形式。


抱歉，我之前理解錯誤了。如果你需要一份講義的章節結構，以下是一個基本的章節和節（section）的建議，以介紹函數式程設（FP）。這是一個簡單的章節結構，你可以根據需要進一步擴展和定制。

**第一章：介紹函數式程設**

1.1 什麼是函數式程設？
   - 程設範式的概念
   - FP的特點和優點

1.2 FP的歷史和演變
   - FP的起源
   - 現代函數式語言

1.3 為什麼學習函數式程設？
   - 提高程式碼品質
   - 支持並行和併發程設
   - 有助於思考和解決問題

**第二章：函數和不可變性**

2.1 函數的基本概念
   - 什麼是函數？
   - 函數的特性：純函數和不純函數

2.2 不可變性
   - 不可變性的意義和好處
   - 不可變資料類型

2.3 高階函數和匿名函數
   - 高階函數的定義
   - Lambda 表達式和匿名函數的使用

**第三章：函數式程設的核心工具**

3.1 map、filter 和 reduce
   - map 函數的應用
   - filter 函數的使用
   - reduce 函數的介紹

3.2 遞歸和遞迴函數
   - 遞歸的基本概念
   - 遞歸函數的設計和使用

**第四章：進階主題**

4.1 閉包和柯里化
   - 閉包的概念和作用
   - 柯里化和部分應用

4.2 模式匹配和模式合一
   - 模式匹配的概念和語法
   - 模式合一的使用

4.3 並行和併發
   - 函數式程設中的並行性
   - 併發程設模型

4.4 Monad 和函數式 I/O
   - Monad的基本理解
   - 函數式I/O的概念

**第五章：函數式語言和工具**

5.1 常見的函數式語言
   - Haskell
   - Scala
   - Clojure

5.2 函數式程設工具和庫
   - 函數庫的選擇和使用
   - 函數式程式設計工具的簡介

這個章節結構可以幫助你組織你的講義，使學生可以按照一個邏輯的流程學習函數式程設的基本概念和進階主題。根據你的需求，你還可以添加更多的章節或節，以更詳細地涵蓋函數式程設的各個方面。