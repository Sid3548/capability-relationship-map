"""Generate widened multi-task batteries as JSONL.

Four tasks:
  coding    -> exec unit tests (pass@1)
  math      -> numeric exact match
  history   -> factual: normalized exact / alias / token-F1
  reasoning -> constrained MCQ (single letter A-D), never free-form

Writing via Python avoids hand-escaping newlines in JSON.
"""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "data" / "batteries"
OUT.mkdir(parents=True, exist_ok=True)


def w(name, rows):
    p = OUT / name
    with open(p, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {p} ({len(rows)} rows)")


# ---------------- CODING (exec) ----------------
def code(id_, entry, prompt_tail, gold, tests):
    return {
        "task": "coding", "id": id_,
        "prompt": f"{prompt_tail}\n\ndef {entry}",
        "gold": gold, "eval_type": "exec", "tests": tests,
        "metadata": {"entry_point": entry},
    }

coding = [
    code("code_01", "factorial(n):",
         "Write a Python function factorial(n) that returns the factorial of a non-negative integer n. Only output the function definition, no explanation.",
         "def factorial(n):\n    r = 1\n    for i in range(2, n + 1):\n        r *= i\n    return r",
         ["assert factorial(0)==1", "assert factorial(1)==1", "assert factorial(5)==120", "assert factorial(10)==3628800"]),
    code("code_02", "is_palindrome(s):",
         "Write a Python function is_palindrome(s) that returns True if string s reads the same forwards and backwards else False. Only output the function definition.",
         "def is_palindrome(s):\n    return s == s[::-1]",
         ["assert is_palindrome('racecar')==True", "assert is_palindrome('hello')==False", "assert is_palindrome('')==True"]),
    code("code_03", "reverse_list(lst):",
         "Write a Python function reverse_list(lst) returning a new list with elements of lst reversed, without using reversed() or list.reverse(). Only output the function definition.",
         "def reverse_list(lst):\n    return lst[::-1]",
         ["assert reverse_list([1,2,3])==[3,2,1]", "assert reverse_list([])==[]", "assert reverse_list([5])==[5]"]),
    code("code_04", "count_vowels(s):",
         "Write a Python function count_vowels(s) returning the number of vowels (aeiou, case-insensitive) in s. Only output the function definition.",
         "def count_vowels(s):\n    return sum(1 for c in s.lower() if c in 'aeiou')",
         ["assert count_vowels('hello')==2", "assert count_vowels('AEIOU')==5", "assert count_vowels('xyz')==0"]),
    code("code_05", "fizzbuzz(n):",
         "Write a Python function fizzbuzz(n) returning a list of strings for 1..n: 'Fizz' if div by 3, 'Buzz' if div by 5, 'FizzBuzz' if both, else str(i). Only output the function definition.",
         "def fizzbuzz(n):\n    o=[]\n    for i in range(1,n+1):\n        if i%15==0: o.append('FizzBuzz')\n        elif i%3==0: o.append('Fizz')\n        elif i%5==0: o.append('Buzz')\n        else: o.append(str(i))\n    return o",
         ["assert fizzbuzz(5)==['1','2','Fizz','4','Buzz']", "assert fizzbuzz(3)==['1','2','Fizz']"]),
    code("code_06", "gcd(a, b):",
         "Write a Python function gcd(a,b) returning the greatest common divisor of two positive integers. Only output the function definition.",
         "def gcd(a, b):\n    while b:\n        a, b = b, a % b\n    return a",
         ["assert gcd(12,8)==4", "assert gcd(17,5)==1", "assert gcd(100,10)==10"]),
    code("code_07", "sum_list(lst):",
         "Write a Python function sum_list(lst) returning the sum of a list of numbers without using the builtin sum(). Only output the function definition.",
         "def sum_list(lst):\n    t = 0\n    for x in lst:\n        t += x\n    return t",
         ["assert sum_list([1,2,3])==6", "assert sum_list([])==0", "assert sum_list([-1,1])==0"]),
    code("code_08", "max_of(lst):",
         "Write a Python function max_of(lst) returning the maximum element of a non-empty list without using builtin max(). Only output the function definition.",
         "def max_of(lst):\n    m = lst[0]\n    for x in lst[1:]:\n        if x > m: m = x\n    return m",
         ["assert max_of([1,5,3])==5", "assert max_of([-2,-9])==-2", "assert max_of([7])==7"]),
    code("code_09", "is_prime(n):",
         "Write a Python function is_prime(n) returning True if n is a prime number else False. Only output the function definition.",
         "def is_prime(n):\n    if n < 2: return False\n    i = 2\n    while i*i <= n:\n        if n % i == 0: return False\n        i += 1\n    return True",
         ["assert is_prime(2)==True", "assert is_prime(15)==False", "assert is_prime(13)==True", "assert is_prime(1)==False"]),
    code("code_10", "fib(n):",
         "Write a Python function fib(n) returning the n-th Fibonacci number with fib(0)=0, fib(1)=1. Only output the function definition.",
         "def fib(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a",
         ["assert fib(0)==0", "assert fib(1)==1", "assert fib(10)==55"]),
    code("code_11", "capitalize_words(s):",
         "Write a Python function capitalize_words(s) that capitalizes the first letter of each space-separated word in s. Only output the function definition.",
         "def capitalize_words(s):\n    return ' '.join(w[:1].upper()+w[1:] for w in s.split(' '))",
         ["assert capitalize_words('hello world')=='Hello World'", "assert capitalize_words('a b')=='A B'"]),
    code("code_12", "count_evens(lst):",
         "Write a Python function count_evens(lst) returning how many elements of lst are even integers. Only output the function definition.",
         "def count_evens(lst):\n    return sum(1 for x in lst if x % 2 == 0)",
         ["assert count_evens([1,2,3,4])==2", "assert count_evens([1,3,5])==0", "assert count_evens([])==0"]),
    code("code_13", "flatten(lst):",
         "Write a Python function flatten(lst) that flattens a list of lists into a single list, one level deep. Only output the function definition.",
         "def flatten(lst):\n    o = []\n    for sub in lst:\n        for x in sub:\n            o.append(x)\n    return o",
         ["assert flatten([[1,2],[3]])==[1,2,3]", "assert flatten([])==[]", "assert flatten([[1]])==[1]"]),
    code("code_14", "second_largest(lst):",
         "Write a Python function second_largest(lst) returning the second largest distinct value in lst (len>=2, at least 2 distinct). Only output the function definition.",
         "def second_largest(lst):\n    u = sorted(set(lst))\n    return u[-2]",
         ["assert second_largest([1,2,3])==2", "assert second_largest([5,5,3])==3", "assert second_largest([9,1])==1"]),
    code("code_15", "char_count(s):",
         "Write a Python function char_count(s) returning a dict mapping each character in s to its count. Only output the function definition.",
         "def char_count(s):\n    d = {}\n    for c in s:\n        d[c] = d.get(c,0)+1\n    return d",
         ["assert char_count('aab')=={'a':2,'b':1}", "assert char_count('')=={}"]),
    code("code_16", "to_snake(s):",
         "Write a Python function to_snake(s) converting a camelCase string to snake_case (lowercase, underscores before original capitals). Only output the function definition.",
         "def to_snake(s):\n    o = ''\n    for c in s:\n        if c.isupper(): o += '_' + c.lower()\n        else: o += c\n    return o",
         ["assert to_snake('camelCase')=='camel_case'", "assert to_snake('abc')=='abc'"]),
]

# ---------------- MATH (numeric) ----------------
def mth(id_, q, gold):
    return {"task": "math", "id": id_, "prompt": f"Q: {q}\nA: The answer is",
            "gold": str(gold), "eval_type": "numeric", "metadata": {"tolerance": 0.0}}

math = [
    mth("math_01", "What is 47 + 58?", 105),
    mth("math_02", "What is 123 - 47?", 76),
    mth("math_03", "What is 12 times 8?", 96),
    mth("math_04", "What is 144 divided by 12?", 12),
    mth("math_05", "If a train travels 60 miles per hour for 3 hours, how many miles does it travel?", 180),
    mth("math_06", "What is 15 percent of 200?", 30),
    mth("math_07", "What is 9 squared?", 81),
    mth("math_08", "What is 250 + 375?", 625),
    mth("math_09", "What is 1000 - 256?", 744),
    mth("math_10", "What is 7 times 13?", 91),
    mth("math_11", "What is 84 divided by 7?", 12),
    mth("math_12", "A shop sells pens at 3 dollars each. How much for 14 pens?", 42),
    mth("math_13", "What is the sum of the first 10 positive integers?", 55),
    mth("math_14", "What is 2 to the power of 10?", 1024),
    mth("math_15", "If 5 apples cost 20 dollars, how much does 1 apple cost?", 4),
    mth("math_16", "What is 360 divided by 8?", 45),
    mth("math_17", "What is 17 + 28 + 5?", 50),
    mth("math_18", "A rectangle is 6 by 9. What is its area?", 54),
]

# ---------------- HISTORY (factual, exact/alias/F1) ----------------
def hist(id_, q, gold, aliases):
    return {"task": "history", "id": id_,
            "prompt": f"Question: {q}\nAnswer:",
            "gold": " " + gold, "eval_type": "factual",
            "metadata": {"aliases": aliases, "f1_threshold": 0.5}}

history = [
    hist("hist_01", "In what year did World War II end?", "1945", ["nineteen forty-five"]),
    hist("hist_02", "Who was the first President of the United States?", "George Washington", ["Washington"]),
    hist("hist_03", "Which country built the Great Wall?", "China", ["ancient China"]),
    hist("hist_04", "In what year did the Berlin Wall fall?", "1989", []),
    hist("hist_05", "Who wrote the Communist Manifesto with Friedrich Engels?", "Karl Marx", ["Marx"]),
    hist("hist_06", "Which ancient civilization built the pyramids at Giza?", "Egypt", ["ancient Egypt", "the Egyptians"]),
    hist("hist_07", "Who was the British Prime Minister during most of World War II?", "Winston Churchill", ["Churchill"]),
    hist("hist_08", "In what year did Christopher Columbus first reach the Americas?", "1492", []),
    hist("hist_09", "Which empire was ruled by Julius Caesar?", "Roman Empire", ["Rome", "the Romans", "Roman"]),
    hist("hist_10", "Who led India's nonviolent independence movement?", "Mahatma Gandhi", ["Gandhi", "Mohandas Gandhi"]),
    hist("hist_11", "In what year did the American Declaration of Independence get signed?", "1776", []),
    hist("hist_12", "Which city was the capital of the Byzantine Empire?", "Constantinople", ["Istanbul"]),
    hist("hist_13", "Who was the female pharaoh known for a famous relationship with Mark Antony?", "Cleopatra", ["Cleopatra VII"]),
    hist("hist_14", "Which war was fought between the North and South of the United States in the 1860s?", "the Civil War", ["American Civil War", "Civil War"]),
    hist("hist_15", "Who painted the Mona Lisa?", "Leonardo da Vinci", ["da Vinci", "Leonardo"]),
    hist("hist_16", "In what year did the Titanic sink?", "1912", []),
]

# ---------------- REASONING (MCQ, single letter) ----------------
def mcq(id_, q, a, b, c, d, gold):
    prompt = (f"Answer with a single letter (A, B, C, or D).\n"
              f"{q}\nA) {a}\nB) {b}\nC) {c}\nD) {d}\nAnswer:")
    return {"task": "reasoning", "id": id_, "prompt": prompt,
            "gold": " " + gold, "eval_type": "mcq",
            "metadata": {"choices": ["A", "B", "C", "D"]}}

reasoning = [
    mcq("reas_01", "If all roses are flowers and some flowers fade quickly, which must be true?",
        "All roses fade quickly", "Some roses may fade quickly", "No roses are flowers", "All flowers are roses", "B"),
    mcq("reas_02", "A is taller than B. B is taller than C. Who is shortest?", "A", "B", "C", "Cannot tell", "C"),
    mcq("reas_03", "What comes next: 2, 4, 8, 16, ?", "18", "24", "32", "20", "C"),
    mcq("reas_04", "If it rains, the ground gets wet. The ground is wet. What can we conclude?",
        "It definitely rained", "It might have rained", "It did not rain", "The ground is dry", "B"),
    mcq("reas_05", "Which number is the odd one out: 3, 5, 8, 11?", "3", "5", "8", "11", "C"),
    mcq("reas_06", "All cats are mammals. Fluffy is a cat. Therefore Fluffy is a:", "bird", "mammal", "fish", "reptile", "B"),
    mcq("reas_07", "What comes next: 1, 1, 2, 3, 5, ?", "6", "7", "8", "9", "C"),
    mcq("reas_08", "If today is Monday, what day is it in 3 days?", "Wednesday", "Thursday", "Friday", "Saturday", "B"),
    mcq("reas_09", "A box has 3 red and 2 blue balls. Odds a random ball is red?", "2 in 5", "3 in 5", "1 in 2", "3 in 2", "B"),
    mcq("reas_10", "Which is heavier: 1 kg of feathers or 1 kg of iron?", "Feathers", "Iron", "They weigh the same", "Cannot tell", "C"),
    mcq("reas_11", "Complete the analogy: hand is to glove as foot is to:", "sock", "hat", "shirt", "ring", "A"),
    mcq("reas_12", "If some birds cannot fly, which is true?", "All birds fly", "No birds fly", "Not all birds fly", "Only birds fly", "C"),
    mcq("reas_13", "What comes next: 100, 90, 80, 70, ?", "65", "60", "50", "75", "B"),
    mcq("reas_14", "Tom is older than Sara. Sara is older than Ben. Who is oldest?", "Ben", "Sara", "Tom", "Cannot tell", "C"),
    mcq("reas_15", "A store gives 50% off. A 40 dollar item now costs:", "10", "20", "30", "35", "B"),
    mcq("reas_16", "Which word does not belong: apple, banana, carrot, cherry?", "apple", "banana", "carrot", "cherry", "C"),
]

w("coding.jsonl", coding)
w("math.jsonl", math)
w("history.jsonl", history)
w("reasoning.jsonl", reasoning)
print("done")
