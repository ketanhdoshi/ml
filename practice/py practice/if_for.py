rangelist = range(10)
print(rangelist)

for number in rangelist:
    # Check if number is one of
    # the numbers in the tuple.
    if number in (3, 4, 7, 9):
         # "Break" terminates a for without
        # executing the "else" clause.
        break
    else:
        print(number)
        # "Continue" starts the next iteration
        # of the loop. It's rather useless here,
        # as it's the last statement of the loop.
        continue
else:
    # The "else" clause is optional and is
    # executed only if the loop didn't "break".
    pass # Do nothing

if rangelist[1] == 2:
    print("The second item (lists are 0-based) is 2")
elif rangelist[1] == 3:
    print("The second item (lists are 0-based) is 3")
else:
    print("Dunno")

lst1 = [1, 2, 3]
lst2 = [3, 4, 5]
print([x * y for x in lst1 for y in lst2])
print([x for x in lst1 if 4 > x > 1])

# Check if a condition is true for any items. "any" returns true if any 
# item in the list is true. Because 4 % 3 = 1, and 1 is true, so any()
# returns True.
print(any([i % 3 for i in [3, 3, 4, 4, 3]]))

# List comprehensions provide a powerful way to create and manipulate lists. 
# They consist of an expression followed by a for clause followed by zero or 
# more if or for clauses
# Check for how many items a condition is true.
print(sum(1 for i in [3, 3, 4, 4, 3] if i == 4))