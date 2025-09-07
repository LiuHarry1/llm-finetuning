import redis

# docker exec -it myredis redis-cli -h 127.0.0.1 -p 6379


r = redis.Redis(host='localhost', port=6379, db=0)
stream_key = "order_stream2"


# 生产者
def produce():
    for i in range(5):
        r.xadd(stream_key, {"order": f"order_{i}"})
        print("生产订单:", f"order_{i}")
        if i ==3:
            print("生产订单:", f"0-0")
            r.xadd(stream_key, {"order": f"0-0"})


# 消费者
def consume():
    last_id = "0-0"
    while True:
        messages = r.xread({stream_key: last_id}, block=2000, count=1)
        if not messages:
            break
        # print("----")
        for _, msgs in messages:
            for msg_id, data in msgs:
                print("消费订单:", msg_id, data)
                last_id = msg_id


produce()
consume()