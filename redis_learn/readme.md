很好 👍，你现在已经能看到 Redis 里有两个 key：

```
1) "order_stream"
2) "testkey"
```

---

## 🔹 查看 key 的数据类型

Redis 提供 `TYPE` 命令来查看 key 的数据类型：

```redis
TYPE order_stream
```

可能返回的结果有：

* `string`
* `list`
* `set`
* `zset`
* `hash`
* `stream`

---

## 🔹 针对不同类型怎么查看内容

1. **string**

```redis
GET order_stream
```

2. **list**

```redis
LRANGE order_stream 0 -1
```

3. **set**

```redis
SMEMBERS order_stream
```

4. **zset（有序集合）**

```redis
ZRANGE order_stream 0 -1 WITHSCORES
```

5. **hash**

```redis
HGETALL order_stream
```

6. **stream**（Redis 5.0+ 新增）

```redis
XRANGE order_stream - +
```

* `- +` 表示从最小 ID 到最大 ID，查看所有消息

---

✅ 所以你下一步可以先执行：

```redis
TYPE order_stream
```

然后根据返回的类型，再用对应命令查看里面的值。

---

要不要我帮你写一个 **Redis 常见数据类型的速查表**，方便你以后快速查看和操作？
