

nums = [1,2,3,4,5]
nums[2:] = nums[2:][::-1]

print(nums)

nums2= nums
nums3 = nums[:]
nums[1] =9
print(nums == nums2)



print(nums2 == nums3)
print(nums2)
print(nums3)