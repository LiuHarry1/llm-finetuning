def count_up_to(n):
    print("Generator started")
    i = 1
    while i <= n:
        yield i     # 返回 i，并暂停
        i += 1
    print("Generator finished")

gen = count_up_to(3)
print(next(gen))  # 1
print(next(gen))  # 2
print(next(gen))  # 3
# print(next(gen))  # 3
# print(next(gen))  # StopIteration 异常
print("stop here======")

for num in count_up_to(5):
    print(num)
