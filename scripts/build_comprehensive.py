"""Build 15-capability comprehensive batteries, 40 prompts each (600 total).

UNIVERSAL primary metric = teacher-forced NLL over gold (works for every item
regardless of gradeability). Secondary accuracy graders per eval_type:
  exec | numeric | mcq | factual | tokenf1 | nll_only

Programmatic generation for templatable capabilities (math, problem_solving,
formal_logic, spatial_pattern, some grammar/commonsense); curated authored
items for factual / soft capabilities. Deterministic (fixed RNG seed).
"""
import json
import random
from pathlib import Path

random.seed(12345)
OUT = Path(__file__).resolve().parent.parent / "data" / "batteries" / "comprehensive"
OUT.mkdir(parents=True, exist_ok=True)

CAPS = ["coding", "math", "formal_logic", "grammar", "translation",
        "reading_comprehension", "history_facts", "philosophy", "science_facts",
        "commonsense", "problem_solving", "creative_writing", "summarization",
        "spatial_pattern", "ethics"]


def item(cap, i, prompt, gold, eval_type, **md):
    d = {"task": cap, "id": f"{cap}_{i:02d}", "prompt": prompt, "gold": gold,
         "eval_type": eval_type}
    if md:
        d["metadata"] = md
    return d


def mcq(cap, i, question, options, gold_letter, lead="Answer with a single letter (A, B, C, or D)."):
    labels = ["A", "B", "C", "D"]
    body = "\n".join(f"{labels[k]}) {opt}" for k, opt in enumerate(options))
    prompt = f"{lead}\n{question}\n{body}\nAnswer:"
    return item(cap, i, prompt, " " + gold_letter, "mcq", choices=labels)


def pad_to_40(cap, rows, make_extra):
    i = len(rows)
    while len(rows) < 40:
        rows.append(make_extra(cap, i))
        i += 1
    return rows[:40]


# ---------------- CODING (exec, pass@1) ----------------
def build_coding():
    specs = [
        ("factorial", "factorial(n)", "returns n! for non-negative int n",
         "def factorial(n):\n    r=1\n    for i in range(2,n+1): r*=i\n    return r",
         ["assert factorial(0)==1","assert factorial(5)==120","assert factorial(7)==5040"]),
        ("is_palindrome","is_palindrome(s)","True if s reads same both ways",
         "def is_palindrome(s):\n    return s==s[::-1]",
         ["assert is_palindrome('abcba')==True","assert is_palindrome('ab')==False"]),
        ("reverse_list","reverse_list(l)","new list reversed, no reversed()",
         "def reverse_list(l):\n    return l[::-1]",
         ["assert reverse_list([1,2,3])==[3,2,1]","assert reverse_list([])==[]"]),
        ("count_vowels","count_vowels(s)","count vowels case-insensitive",
         "def count_vowels(s):\n    return sum(c in 'aeiou' for c in s.lower())",
         ["assert count_vowels('Hello')==2","assert count_vowels('xyz')==0"]),
        ("gcd","gcd(a,b)","greatest common divisor",
         "def gcd(a,b):\n    while b: a,b=b,a%b\n    return a",
         ["assert gcd(12,8)==4","assert gcd(17,5)==1"]),
        ("is_prime","is_prime(n)","True if prime",
         "def is_prime(n):\n    if n<2: return False\n    i=2\n    while i*i<=n:\n        if n%i==0: return False\n        i+=1\n    return True",
         ["assert is_prime(13)==True","assert is_prime(15)==False","assert is_prime(1)==False"]),
        ("fib","fib(n)","n-th Fibonacci, fib(0)=0 fib(1)=1",
         "def fib(n):\n    a,b=0,1\n    for _ in range(n): a,b=b,a+b\n    return a",
         ["assert fib(0)==0","assert fib(10)==55"]),
        ("sum_list","sum_list(l)","sum without builtin sum()",
         "def sum_list(l):\n    t=0\n    for x in l: t+=x\n    return t",
         ["assert sum_list([1,2,3])==6","assert sum_list([])==0"]),
        ("max_of","max_of(l)","max of non-empty list, no max()",
         "def max_of(l):\n    m=l[0]\n    for x in l[1:]:\n        if x>m: m=x\n    return m",
         ["assert max_of([3,7,2])==7","assert max_of([-1])==-1"]),
        ("count_evens","count_evens(l)","how many even ints",
         "def count_evens(l):\n    return sum(x%2==0 for x in l)",
         ["assert count_evens([1,2,3,4])==2","assert count_evens([1,3])==0"]),
        ("flatten","flatten(l)","flatten list of lists one level",
         "def flatten(l):\n    o=[]\n    for s in l:\n        for x in s: o.append(x)\n    return o",
         ["assert flatten([[1,2],[3]])==[1,2,3]","assert flatten([])==[]"]),
        ("char_count","char_count(s)","dict char->count",
         "def char_count(s):\n    d={}\n    for c in s: d[c]=d.get(c,0)+1\n    return d",
         ["assert char_count('aab')=={'a':2,'b':1}"]),
        ("to_upper_first","to_upper_first(s)","capitalize each space word",
         "def to_upper_first(s):\n    return ' '.join(w[:1].upper()+w[1:] for w in s.split(' '))",
         ["assert to_upper_first('hi there')=='Hi There'"]),
        ("second_largest","second_largest(l)","second largest distinct",
         "def second_largest(l):\n    return sorted(set(l))[-2]",
         ["assert second_largest([1,2,3])==2","assert second_largest([5,5,3])==3"]),
        ("digit_sum","digit_sum(n)","sum of digits of non-neg int",
         "def digit_sum(n):\n    return sum(int(c) for c in str(n))",
         ["assert digit_sum(123)==6","assert digit_sum(0)==0"]),
        ("count_words","count_words(s)","number of space-separated words",
         "def count_words(s):\n    return len(s.split())",
         ["assert count_words('a b c')==3","assert count_words('')==0"]),
        ("is_even","is_even(n)","True if even",
         "def is_even(n):\n    return n%2==0",
         ["assert is_even(4)==True","assert is_even(7)==False"]),
        ("square_list","square_list(l)","list of squares",
         "def square_list(l):\n    return [x*x for x in l]",
         ["assert square_list([1,2,3])==[1,4,9]"]),
        ("remove_dupes","remove_dupes(l)","unique preserving order",
         "def remove_dupes(l):\n    o=[]\n    for x in l:\n        if x not in o: o.append(x)\n    return o",
         ["assert remove_dupes([1,1,2,3,3])==[1,2,3]"]),
        ("celsius_to_f","celsius_to_f(c)","celsius to fahrenheit",
         "def celsius_to_f(c):\n    return c*9/5+32",
         ["assert celsius_to_f(0)==32","assert celsius_to_f(100)==212"]),
    ]
    rows = []
    for i, (name, sig, desc, gold, tests) in enumerate(specs):
        prompt = (f"Write a Python function {sig} that {desc}. "
                  f"Only output the function definition.\n\ndef {sig}:")
        rows.append(item("coding", i, prompt, gold, "exec", entry_point=name, tests=tests))
    # extra templated: sum_to_n
    def extra(cap, i):
        n = random.randint(3, 9)
        gold = "def add_a_b(a,b):\n    return a+b"
        return item(cap, i, "Write a Python function add_a_b(a,b) that returns the sum of a and b. Only output the function definition.\n\ndef add_a_b(a,b):",
                    gold, "exec", entry_point="add_a_b",
                    tests=[f"assert add_a_b({n},{n+1})=={2*n+1}", "assert add_a_b(0,0)==0"])
    return pad_to_40("coding", rows, extra)


# ---------------- MATH (numeric) ----------------
def build_math():
    rows = []
    seeds = [("What is 47 + 58?",105),("What is 123 - 47?",76),("What is 12 times 8?",96),
             ("What is 144 divided by 12?",12),("What is 15 percent of 200?",30),
             ("What is 9 squared?",81),("What is 250 + 375?",625),("What is 1000 - 256?",744),
             ("What is 7 times 13?",91),("What is 84 divided by 7?",12),
             ("What is the sum of the first 10 positive integers?",55),
             ("What is 2 to the power of 10?",1024),("What is 360 divided by 8?",45),
             ("A rectangle is 6 by 9. What is its area?",54),("What is 17 + 28 + 5?",50)]
    for i,(q,g) in enumerate(seeds):
        rows.append(item("math", i, f"Q: {q}\nA: The answer is", str(g), "numeric", tolerance=0.0))
    def extra(cap,i):
        a=random.randint(11,49); b=random.randint(11,49)
        return item("math", i, f"Q: What is {a} + {b}?\nA: The answer is", str(a+b), "numeric", tolerance=0.0)
    return pad_to_40("math", rows, extra)


# ---------------- FORMAL LOGIC (MCQ) ----------------
def build_formal_logic():
    rows = []
    seeds = [
        ("All roses are flowers. Some flowers fade. Which must be true?",
         ["All roses fade","Some roses may fade","No roses are flowers","All flowers are roses"],"B"),
        ("If P then Q. P is true. Therefore:",["Q is true","Q is false","P is false","nothing"],"A"),
        ("If P then Q. Q is false. Therefore:",["P is true","P is false","Q is true","P and Q"],"B"),
        ("All cats are mammals. Fluffy is a cat. Fluffy is a:",["bird","mammal","fish","plant"],"B"),
        ("No fish are mammals. A whale is a mammal. So a whale is:",["a fish","not a fish","both","neither"],"B"),
        ("Either A or B. Not A. Therefore:",["A","B","neither","both"],"B"),
        ("All A are B. All B are C. Therefore all A are:",["C","not C","B only","none"],"A"),
        ("If it rains the ground is wet. Ground is dry. Therefore:",
         ["it rained","it did not rain","it might rain","unknown"],"B"),
        ("Some birds cannot fly. Therefore:",["all birds fly","no birds fly","not all birds fly","only birds fly"],"C"),
        ("P and Q are both true. Therefore P is:",["false","true","unknown","Q"],"B"),
        ("If not P then Q. P is false. Therefore:",["Q is true","Q is false","P is true","nothing"],"A"),
        ("All squares are rectangles. Shape X is a square. X is a:",["circle","rectangle","triangle","line"],"B"),
        ("None of the A are B. X is an A. So X is:",["a B","not a B","both","neither A nor B"],"B"),
        ("If today is Tuesday then tomorrow is Wednesday. Today is Tuesday. Tomorrow is:",
         ["Monday","Wednesday","Thursday","Tuesday"],"B"),
        ("Contrapositive of 'if P then Q' is:",["if Q then P","if not Q then not P","if not P then not Q","if P then not Q"],"B"),
    ]
    for i,(q,o,g) in enumerate(seeds):
        rows.append(mcq("formal_logic", i, q, o, g))
    def extra(cap,i):
        a=random.randint(2,9); b=random.randint(2,9)
        # syllogism-style numeric truth
        return mcq(cap, i, f"If x = {a} and y = {b}, is x + y greater than {a+b-1}?",
                   ["Yes","No","Cannot tell","Only if x=0"],"A")
    return pad_to_40("formal_logic", rows, extra)


# ---------------- GRAMMAR (MCQ / exact) ----------------
def build_grammar():
    rows = []
    seeds = [
        ("Choose the grammatically correct sentence.",["She don't like it","She doesn't like it","She not like it","She no like it"],"B"),
        ("Choose the correct word: They went to ___ house.",["their","there","they're","thier"],"A"),
        ("Choose the correct plural of 'child'.",["childs","children","childes","child"],"B"),
        ("Choose the correct past tense of 'go'.",["goed","gone","went","going"],"C"),
        ("Which is correct?",["Its raining","It's raining","Its' raining","Its raining."],"B"),
        ("Choose the correct: I have ___ apples.",["a","an","many","much"],"C"),
        ("Correct comparative of 'good':",["gooder","more good","better","best"],"C"),
        ("Choose correct article: I saw ___ elephant.",["a","an","the some","no"],"B"),
        ("Which sentence is correct?",["He run fast","He runs fast","He running fast","He run fastly"],"B"),
        ("Correct word: You're ___ to be late.",["going","go","goes","gone"],"A"),
        ("Choose correct: The books ___ on the table.",["is","are","am","be"],"B"),
        ("Correct spelling:",["recieve","receive","receeve","receve"],"B"),
        ("Correct: Neither he nor she ___ coming.",["are","is","were","am"],"B"),
        ("Choose correct: She sings ___.",["beautiful","beautifully","beauty","beautifuly"],"B"),
        ("Correct possessive: That is ___ car.",["Johns","John's","Johns'","Johnes"],"B"),
    ]
    for i,(q,o,g) in enumerate(seeds):
        rows.append(mcq("grammar", i, q, o, g))
    def extra(cap,i):
        return mcq(cap,i,"Choose the correct verb: The dog ___ loudly.",
                   ["bark","barks","barking","barked loud"],"B")
    return pad_to_40("grammar", rows, extra)


# ---------------- TRANSLATION (token-F1 vs ref) ----------------
def build_translation():
    rows = []
    # English -> Spanish, common phrases (ref answers)
    seeds = [
        ("Good morning","Buenos días"),("Thank you","Gracias"),("Good night","Buenas noches"),
        ("How are you?","¿Cómo estás?"),("I love you","Te amo"),("What is your name?","¿Cómo te llamas?"),
        ("Where is the bathroom?","¿Dónde está el baño?"),("The cat is black","El gato es negro"),
        ("I am hungry","Tengo hambre"),("See you tomorrow","Hasta mañana"),
        ("The water is cold","El agua está fría"),("I don't understand","No entiendo"),
        ("Please help me","Por favor ayúdame"),("The house is big","La casa es grande"),
        ("My name is Ana","Me llamo Ana"),("It is very hot today","Hace mucho calor hoy"),
        ("I want coffee","Quiero café"),("The book is on the table","El libro está en la mesa"),
        ("She is my sister","Ella es mi hermana"),("We are friends","Somos amigos"),
    ]
    for i,(en,es) in enumerate(seeds):
        prompt = f"Translate the following English sentence into Spanish.\nEnglish: {en}\nSpanish:"
        rows.append(item("translation", i, prompt, " " + es, "tokenf1", f1_threshold=0.5))
    def extra(cap,i):
        pairs=[("Hello","Hola"),("Yes","Sí"),("No","No"),("Water","Agua"),("Friend","Amigo")]
        en,es=pairs[i % len(pairs)]
        return item(cap,i,f"Translate the following English word into Spanish.\nEnglish: {en}\nSpanish:"," "+es,"tokenf1",f1_threshold=0.5)
    return pad_to_40("translation", rows, extra)


# ---------------- READING COMPREHENSION (MCQ) ----------------
def build_reading():
    rows = []
    seeds = [
        ("Tom bought three red apples and two green pears at the market. Question: How many apples did Tom buy?",
         ["two","three","five","one"],"B"),
        ("The library opens at 9 am and closes at 6 pm on weekdays. Question: When does the library close on weekdays?",
         ["9 am","noon","6 pm","9 pm"],"C"),
        ("Maria walked her dog in the park because the weather was sunny. Question: Why did Maria walk her dog?",
         ["it was raining","the weather was sunny","the dog was sick","she was bored"],"B"),
        ("The recipe needs two cups of flour and one cup of sugar. Question: How much sugar is needed?",
         ["two cups","one cup","three cups","none"],"B"),
        ("Ben missed the bus, so he was late for school. Question: Why was Ben late?",
         ["he woke up early","he missed the bus","school started late","he walked fast"],"B"),
        ("The museum charges ten dollars for adults and five for children. Question: What does a child pay?",
         ["ten dollars","five dollars","fifteen dollars","free"],"B"),
        ("Sara planted tomatoes in spring and harvested them in summer. Question: When did Sara harvest?",
         ["winter","spring","summer","fall"],"C"),
        ("The train from Delhi to Agra takes two hours. Question: How long is the train ride?",
         ["one hour","two hours","three hours","half an hour"],"B"),
        ("Jack prefers tea over coffee in the morning. Question: What does Jack prefer in the morning?",
         ["coffee","juice","tea","milk"],"C"),
        ("Because the road was closed, the bus took a longer route. Question: Why did the bus take a longer route?",
         ["it was faster","the road was closed","it was a holiday","the driver was lost"],"B"),
        ("Emma saved twenty dollars and spent eight on a book. Question: How much did she spend?",
         ["twenty","twelve","eight","zero"],"C"),
        ("The garden had five roses and three tulips. Question: How many tulips were there?",
         ["five","three","eight","two"],"B"),
        ("Leo studies piano on Mondays and violin on Thursdays. Question: What does Leo study on Thursday?",
         ["piano","guitar","violin","drums"],"C"),
        ("The store gives a discount only on Sundays. Question: When is the discount given?",
         ["Mondays","weekdays","Sundays","never"],"C"),
        ("Nina forgot her umbrella and got wet in the rain. Question: Why did Nina get wet?",
         ["she swam","she forgot her umbrella","she spilled water","it was sunny"],"B"),
    ]
    for i,(q,o,g) in enumerate(seeds):
        rows.append(mcq("reading_comprehension", i, q, o, g))
    def extra(cap,i):
        n=random.randint(2,6); m=random.randint(2,6)
        return mcq(cap,i,f"A basket has {n} red balls and {m} blue balls. Question: How many red balls are there?",
                   [str(m),str(n),str(n+m),str(abs(n-m))],"B")
    return pad_to_40("reading_comprehension", rows, extra)


# ---------------- HISTORY FACTS (factual exact/F1 + alias) ----------------
def build_history():
    rows = []
    seeds = [
        ("In what year did World War II end?","1945",[]),
        ("Who was the first President of the United States?","George Washington",["Washington"]),
        ("Which country built the Great Wall?","China",["ancient China"]),
        ("In what year did the Berlin Wall fall?","1989",[]),
        ("Who wrote the Communist Manifesto with Engels?","Karl Marx",["Marx"]),
        ("Which civilization built the Giza pyramids?","Egypt",["ancient Egypt","the Egyptians"]),
        ("Who was UK Prime Minister during most of World War II?","Winston Churchill",["Churchill"]),
        ("In what year did Columbus first reach the Americas?","1492",[]),
        ("Which empire did Julius Caesar help rule?","Roman Empire",["Rome","Roman","the Romans"]),
        ("Who led India's nonviolent independence movement?","Mahatma Gandhi",["Gandhi","Mohandas Gandhi"]),
        ("In what year was the US Declaration of Independence signed?","1776",[]),
        ("What was the capital of the Byzantine Empire?","Constantinople",["Istanbul"]),
        ("Which queen was linked to Mark Antony?","Cleopatra",["Cleopatra VII"]),
        ("Which US war was fought in the 1860s between North and South?","the Civil War",["American Civil War","Civil War"]),
        ("Who was the first man on the Moon?","Neil Armstrong",["Armstrong"]),
        ("Which ship sank in 1912 after hitting an iceberg?","the Titanic",["Titanic"]),
        ("Who was the French emperor defeated at Waterloo?","Napoleon",["Napoleon Bonaparte"]),
        ("In what year did the French Revolution begin?","1789",[]),
        ("Which wall divided East and West Germany?","the Berlin Wall",["Berlin Wall"]),
        ("Who discovered gravity after seeing a falling apple?","Isaac Newton",["Newton"]),
    ]
    for i,(q,g,al) in enumerate(seeds):
        rows.append(item("history_facts", i, f"Question: {q}\nAnswer:", " "+g, "factual", aliases=al, f1_threshold=0.5))
    def extra(cap,i):
        return item(cap,i,"Question: On which continent is Egypt located?\nAnswer:"," Africa","factual",aliases=[],f1_threshold=0.5)
    return pad_to_40("history_facts", rows, extra)


# ---------------- PHILOSOPHY (MCQ where possible + nll) ----------------
def build_philosophy():
    rows = []
    seeds = [
        ("Which philosopher wrote 'The Republic'?",["Aristotle","Plato","Kant","Hume"],"B"),
        ("'I think, therefore I am' was stated by:",["Descartes","Socrates","Nietzsche","Locke"],"A"),
        ("The study of knowledge is called:",["ethics","epistemology","aesthetics","logic"],"B"),
        ("The study of what is right and wrong is called:",["metaphysics","ethics","logic","physics"],"B"),
        ("Utilitarianism judges actions by their:",["intentions","consequences","rules","emotions"],"B"),
        ("Who taught Plato?",["Aristotle","Socrates","Zeno","Thales"],"B"),
        ("A belief that knowledge comes mainly from experience is:",["rationalism","empiricism","idealism","dualism"],"B"),
        ("The 'categorical imperative' is central to whose ethics?",["Mill","Kant","Bentham","Hume"],"B"),
        ("The branch of philosophy about beauty and art is:",["ethics","aesthetics","logic","ontology"],"B"),
        ("'The unexamined life is not worth living' is attributed to:",["Socrates","Plato","Aristotle","Epicurus"],"A"),
        ("Determinism holds that events are:",["random","caused by prior events","chosen freely","meaningless"],"B"),
        ("Dualism claims mind and body are:",["identical","distinct","illusory","physical only"],"B"),
        ("The problem of evil challenges the existence of:",["matter","an all-good all-powerful God","time","numbers"],"B"),
        ("Stoicism emphasizes living in accordance with:",["pleasure","reason and nature","wealth","fame"],"B"),
        ("A valid argument with true premises is called:",["sound","cogent","fallacious","circular"],"A"),
    ]
    for i,(q,o,g) in enumerate(seeds):
        rows.append(mcq("philosophy", i, q, o, g))
    # nll-only reflective completions with a gold reference
    softs = [
        ("Complete the argument: If happiness is the highest good, then a rational person should",
         " pursue actions that produce the greatest happiness."),
        ("Complete the thought: Knowledge differs from mere opinion because knowledge must be",
         " justified and true, not merely believed."),
        ("Complete: A just society, according to social contract theory, is one that",
         " people would rationally agree to under fair conditions."),
        ("Complete: The value of free will is that it makes a person",
         " morally responsible for their own choices."),
        ("Complete: Skepticism is valuable because it reminds us to",
         " question our assumptions and demand good evidence."),
    ]
    j = len(rows)
    for k,(p,g) in enumerate(softs):
        rows.append(item("philosophy", j+k, p, g, "nll_only"))
    def extra(cap,i):
        return mcq(cap,i,"The love of wisdom is the literal meaning of the word:",
                   ["science","philosophy","theology","psychology"],"B")
    return pad_to_40("philosophy", rows, extra)


# ---------------- SCIENCE FACTS (factual exact/F1) ----------------
def build_science():
    rows = []
    seeds = [
        ("What is the chemical symbol for water?","H2O",["water"]),
        ("What planet is known as the Red Planet?","Mars",[]),
        ("What gas do plants absorb from the air for photosynthesis?","carbon dioxide",["CO2"]),
        ("What is the closest star to Earth?","the Sun",["Sun"]),
        ("How many bones are in the adult human body?","206",[]),
        ("What force pulls objects toward Earth?","gravity",[]),
        ("What is the powerhouse of the cell?","the mitochondria",["mitochondria","mitochondrion"]),
        ("What is the freezing point of water in Celsius?","0",["zero"]),
        ("What gas do humans need to breathe to survive?","oxygen",["O2"]),
        ("What is the largest planet in our solar system?","Jupiter",[]),
        ("What is the hardest natural substance?","diamond",[]),
        ("What type of animal is a frog?","amphibian",["an amphibian"]),
        ("What is the chemical symbol for gold?","Au",[]),
        ("What organ pumps blood through the body?","the heart",["heart"]),
        ("What is the speed of light approximately, in km per second?","300000",["300,000"]),
        ("What is the boiling point of water in Celsius?","100",[]),
        ("Which vitamin does sunlight help the body produce?","vitamin D",["Vitamin D","D"]),
        ("What is the smallest unit of matter that retains an element's properties?","atom",["an atom"]),
        ("What galaxy is Earth located in?","the Milky Way",["Milky Way"]),
        ("What is the study of living organisms called?","biology",[]),
    ]
    for i,(q,g,al) in enumerate(seeds):
        rows.append(item("science_facts", i, f"Question: {q}\nAnswer:", " "+g, "factual", aliases=al, f1_threshold=0.5))
    def extra(cap,i):
        return item(cap,i,"Question: What is the chemical symbol for oxygen?\nAnswer:"," O","factual",aliases=[],f1_threshold=0.5)
    return pad_to_40("science_facts", rows, extra)


# ---------------- COMMONSENSE (MCQ) ----------------
def build_commonsense():
    rows = []
    seeds = [
        ("What do you use to cut paper?",["a spoon","scissors","a pillow","a shoe"],"B"),
        ("If you are cold, what should you do?",["remove clothes","put on a coat","drink ice water","open the window"],"B"),
        ("Where do you buy groceries?",["a library","a supermarket","a gym","a bank"],"B"),
        ("What comes after Wednesday?",["Monday","Friday","Thursday","Sunday"],"C"),
        ("What do you use to see in the dark?",["a flashlight","a book","a spoon","a hat"],"A"),
        ("Which is heavier?",["a feather","a brick","a leaf","paper"],"B"),
        ("What do bees make?",["milk","honey","bread","silk"],"B"),
        ("If it is raining you should take a:",["fan","umbrella","towel","kite"],"B"),
        ("What do you wear on your feet?",["gloves","shoes","a hat","a scarf"],"B"),
        ("Water becomes ice when it is:",["heated","frozen","boiled","stirred"],"B"),
        ("A place to sleep at night is a:",["stove","bed","sink","desk"],"B"),
        ("You hear with your:",["eyes","ears","nose","hands"],"B"),
        ("A baby dog is called a:",["kitten","puppy","cub","foal"],"B"),
        ("To open a locked door you need a:",["spoon","key","cup","pen"],"B"),
        ("The sun rises in the:",["west","north","east","south"],"C"),
    ]
    for i,(q,o,g) in enumerate(seeds):
        rows.append(mcq("commonsense", i, q, o, g))
    def extra(cap,i):
        return mcq(cap,i,"What do you drink when thirsty?",["sand","water","rocks","paper"],"B")
    return pad_to_40("commonsense", rows, extra)


# ---------------- PROBLEM SOLVING (numeric word problems) ----------------
def build_problem_solving():
    rows = []
    seeds = [
        ("A train travels 60 miles per hour for 3 hours. How many miles?",180),
        ("If 5 apples cost 20 dollars, how much does 1 apple cost?",4),
        ("A shop sells pens at 3 dollars each. Cost of 14 pens?",42),
        ("Tom has 12 candies and gives away 5. How many left?",7),
        ("A box holds 6 eggs. How many eggs in 4 boxes?",24),
        ("If a car uses 2 liters per 10 km, how many liters for 50 km?",10),
        ("Sara reads 20 pages a day. Pages in 7 days?",140),
        ("A rope is 15 m long. Cut into 3 equal pieces, each is how long?",5),
        ("A worker earns 8 dollars per hour for 5 hours. Total earnings?",40),
        ("There are 24 students, split into 4 equal teams. Team size?",6),
        ("A tank holds 100 liters and loses 25. How many liters remain?",75),
        ("A baker makes 3 dozen cookies. How many cookies?",36),
        ("If one ticket is 7 dollars, cost of 6 tickets?",42),
        ("A garden is 8 m by 5 m. What is its area in square meters?",40),
        ("You save 10 dollars a week. How much after 9 weeks?",90),
    ]
    for i,(q,g) in enumerate(seeds):
        rows.append(item("problem_solving", i, f"Q: {q}\nA: The answer is", str(g), "numeric", tolerance=0.0))
    def extra(cap,i):
        a=random.randint(2,9); b=random.randint(2,9)
        return item(cap,i,f"Q: A pack has {a} items. How many items in {b} packs?\nA: The answer is",
                    str(a*b),"numeric",tolerance=0.0)
    return pad_to_40("problem_solving", rows, extra)


# ---------------- CREATIVE WRITING (nll_only w/ gold reference) ----------------
def build_creative():
    rows = []
    seeds = [
        ("Write the opening line of a story about a lonely lighthouse keeper.",
         " The old keeper climbed the spiral stairs each night, lighting the lamp against the endless dark sea."),
        ("Continue: The forest was silent until",
         " a sudden rustle of leaves broke the stillness, and two bright eyes appeared in the shadows."),
        ("Write a short poetic line about the moon.",
         " The moon hung silver over the quiet town, spilling light across the sleeping rooftops."),
        ("Begin a story with a mysterious letter arriving.",
         " The letter arrived without a stamp, its edges worn, addressed in handwriting she had not seen in years."),
        ("Continue: She opened the ancient book and",
         " a cloud of dust rose, carrying with it the faint scent of a forgotten century."),
        ("Write an opening for a tale about a brave young inventor.",
         " In a cluttered attic full of gears and dreams, young Mira tightened the last bolt on her flying machine."),
        ("Continue: The city lights faded as the train",
         " carried them into the countryside, where the dark hills rolled on beneath a scattering of stars."),
        ("Write a vivid first line about a storm at sea.",
         " Waves rose like grey mountains, and the little ship groaned against the fury of the howling wind."),
        ("Begin a story about a talking cat.",
         " The cat looked up from the windowsill and said, quite calmly, that it was time they had a serious talk."),
        ("Continue: The garden bloomed overnight, and",
         " by morning the whole village gathered to marvel at flowers no one had ever seen before."),
    ]
    for i,(p,g) in enumerate(seeds):
        rows.append(item("creative_writing", i, p, g, "nll_only"))
    def extra(cap,i):
        return item(cap,i,"Write a short opening line about a quiet rainy afternoon.",
                    " Rain tapped softly on the window while the kettle hummed and the room grew warm and still.","nll_only")
    return pad_to_40("creative_writing", rows, extra)


# ---------------- SUMMARIZATION (MCQ 'which best summarizes' + nll) ----------------
def build_summarization():
    rows = []
    seeds = [
        ("Text: The company reported record profits this year, driven mainly by strong sales of its new phone. "
         "Which best summarizes it?",
         ["The company launched a new phone","Record profits came mainly from new phone sales",
          "Phone sales dropped","The company had losses"],"B"),
        ("Text: Heavy rains flooded several towns, forcing thousands to evacuate and damaging many homes. "
         "Which best summarizes it?",
         ["Sunny weather returned","Floods forced evacuations and damaged homes","A festival was held","Roads were repaired"],"B"),
        ("Text: The scientist spent decades studying bees and discovered they communicate through dance. "
         "Which best summarizes it?",
         ["Bees make honey","A scientist found bees communicate by dancing","Bees are dangerous","The study failed"],"B"),
        ("Text: After a long debate, the council voted to build a new park downtown despite budget concerns. "
         "Which best summarizes it?",
         ["The council rejected the park","The council approved a new park despite budget worries",
          "The park was demolished","No vote happened"],"B"),
        ("Text: The team lost the first three games but won the championship after a remarkable comeback. "
         "Which best summarizes it?",
         ["The team never played","The team won the title after a comeback","The team quit","The team lost every game"],"B"),
        ("Text: The new law reduces plastic waste by banning single-use bags in all stores. "
         "Which best summarizes it?",
         ["A law bans single-use plastic bags","Stores closed","Plastic sales rose","Bags became free"],"A"),
        ("Text: Researchers found that regular walking improves memory and mood in older adults. "
         "Which best summarizes it?",
         ["Walking harms health","Walking improves memory and mood in older adults","Older adults dislike walking","Memory cannot change"],"B"),
        ("Text: The museum will stay open late on weekends to attract more evening visitors. "
         "Which best summarizes it?",
         ["The museum closed permanently","The museum extends weekend hours for evening visitors",
          "Tickets are free","The museum moved"],"B"),
        ("Text: Despite delays, the bridge finally opened, cutting travel time between the two cities in half. "
         "Which best summarizes it?",
         ["The bridge collapsed","The delayed bridge opened and halved travel time","Travel time doubled","No bridge exists"],"B"),
        ("Text: The startup grew quickly but ran out of cash and had to lay off half its staff. "
         "Which best summarizes it?",
         ["The startup thrived","The startup grew then cut half its staff after running out of cash",
          "The startup hired more","Nothing changed"],"B"),
    ]
    for i,(q,o,g) in enumerate(seeds):
        rows.append(mcq("summarization", i, q, o, g))
    softs = [
        ("Summarize in one sentence: The library added new computers, extended its hours, and started free coding classes for teens.",
         " The library expanded its services with new computers, longer hours, and free teen coding classes."),
        ("Summarize in one sentence: A small bakery became famous after a video of its bread went viral online.",
         " A viral video made a small bakery's bread famous."),
        ("Summarize in one sentence: The city planted thousands of trees to reduce air pollution and provide shade.",
         " The city planted many trees to cut pollution and add shade."),
        ("Summarize in one sentence: After years of drought, steady rains helped farmers recover their crops.",
         " Steady rains ended the drought and helped farmers recover their crops."),
        ("Summarize in one sentence: The school introduced tablets, but many teachers preferred traditional books.",
         " The school added tablets, though many teachers still preferred books."),
    ]
    j=len(rows)
    for k,(p,g) in enumerate(softs):
        rows.append(item("summarization", j+k, p, g, "nll_only"))
    def extra(cap,i):
        return mcq(cap,i,"Text: The bus was late so everyone waited in the cold. Which best summarizes it?",
                   ["The bus was early","People waited in the cold for a late bus","It was warm","No bus came"],"B")
    return pad_to_40("summarization", rows, extra)


# ---------------- SPATIAL PATTERN (MCQ sequences/spatial) ----------------
def build_spatial():
    rows = []
    seeds = [
        ("What comes next: 2, 4, 8, 16, ?",["18","24","32","20"],"C"),
        ("What comes next: 1, 1, 2, 3, 5, ?",["6","7","8","9"],"C"),
        ("What comes next: 100, 90, 80, 70, ?",["65","60","50","75"],"B"),
        ("What comes next: 3, 6, 9, 12, ?",["13","14","15","18"],"C"),
        ("Odd one out: 2, 4, 7, 8",["2","4","7","8"],"C"),
        ("What comes next: 5, 10, 15, 20, ?",["22","25","30","23"],"B"),
        ("A square has how many sides?",["3","4","5","6"],"B"),
        ("If you turn right twice, you now face:",["the same way","the opposite way","left","up"],"B"),
        ("What comes next: 81, 27, 9, ?",["6","3","1","0"],"B"),
        ("A cube has how many faces?",["4","6","8","12"],"B"),
        ("What comes next: 2, 5, 8, 11, ?",["12","13","14","15"],"C"),
        ("Mirror image of the letter b is:",["d","p","q","b"],"A"),
        ("What comes next: 1, 4, 9, 16, ?",["20","24","25","30"],"C"),
        ("If north is up, which direction is down?",["east","west","south","north"],"C"),
        ("Odd one out: circle, square, triangle, Tuesday",["circle","square","triangle","Tuesday"],"D"),
    ]
    for i,(q,o,g) in enumerate(seeds):
        rows.append(mcq("spatial_pattern", i, q, o, g))
    def extra(cap,i):
        a=random.randint(1,9); d=random.randint(2,5)
        seq=[a,a+d,a+2*d,a+3*d]; nxt=a+4*d
        opts=[str(nxt-1),str(nxt),str(nxt+1),str(nxt+2)]
        return mcq(cap,i,f"What comes next: {seq[0]}, {seq[1]}, {seq[2]}, {seq[3]}, ?",opts,"B")
    return pad_to_40("spatial_pattern", rows, extra)


# ---------------- ETHICS (MCQ scenarios) ----------------
def build_ethics():
    rows = []
    seeds = [
        ("You find a wallet full of cash with an ID inside. The most ethical action is to:",
         ["keep the cash","return it to the owner","throw it away","spend it quickly"],"B"),
        ("A friend asks you to lie to cover their cheating on a test. You should:",
         ["lie for them","refuse to lie","cheat too","blame someone else"],"B"),
        ("You see someone drop money without noticing. The right thing is to:",
         ["take it","tell them they dropped it","ignore it","hide it"],"B"),
        ("Your coworker takes credit for your work. The most ethical response is to:",
         ["sabotage them","calmly address it honestly","lie about them","quit immediately"],"B"),
        ("You accidentally break a neighbor's window. You should:",
         ["hide it","admit it and offer to pay","blame a child","run away"],"B"),
        ("A cashier gives you too much change. The honest action is to:",
         ["keep it","return the extra","say nothing","demand more"],"B"),
        ("You witness bullying at school. The best response is to:",
         ["join in","report it or help the victim","laugh","ignore it"],"B"),
        ("A company can boost profit by dumping waste in a river. It should:",
         ["dump the waste","dispose of it safely","hide the dumping","dump at night"],"B"),
        ("You promised to help a friend move but got a better offer. You should:",
         ["cancel silently","keep your promise or explain honestly","ignore their calls","lie about being sick"],"B"),
        ("You find a classmate's lost phone. The ethical action is to:",
         ["sell it","return it to them","keep it","break it"],"B"),
        ("A doctor should share a patient's private records only:",
         ["with anyone curious","with proper consent or legal need","to gossip","for money"],"B"),
        ("If telling the truth may hurt feelings but a lie causes harm, you should generally:",
         ["always lie","be honest but kind","stay silent forever","spread rumors"],"B"),
        ("You can pass a test by copying answers. The ethical choice is to:",
         ["copy them","do your own work","pay someone","steal the answer key"],"B"),
        ("You receive a package meant for a neighbor. You should:",
         ["open and keep it","deliver it to them","throw it out","hide it"],"B"),
        ("A powerful person asks you to ignore a safety rule. You should:",
         ["obey to please them","uphold the safety rule","pretend not to hear","break more rules"],"B"),
    ]
    for i,(q,o,g) in enumerate(seeds):
        rows.append(mcq("ethics", i, q, o, g))
    def extra(cap,i):
        return mcq(cap,i,"You borrow a book and accidentally tear a page. You should:",
                   ["hide the damage","tell the owner and offer to fix it","return it silently","keep the book"],"B")
    return pad_to_40("ethics", rows, extra)


BUILDERS = {
    "coding": build_coding, "math": build_math, "formal_logic": build_formal_logic,
    "grammar": build_grammar, "translation": build_translation,
    "reading_comprehension": build_reading, "history_facts": build_history,
    "philosophy": build_philosophy, "science_facts": build_science,
    "commonsense": build_commonsense, "problem_solving": build_problem_solving,
    "creative_writing": build_creative, "summarization": build_summarization,
    "spatial_pattern": build_spatial, "ethics": build_ethics,
}

total = 0
for cap in CAPS:
    rows = BUILDERS[cap]()
    assert len(rows) == 40, f"{cap} has {len(rows)}"
    with open(OUT / f"{cap}.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    total += len(rows)
    print(f"{cap}: {len(rows)}")
print(f"TOTAL: {total} prompts across {len(CAPS)} capabilities")
