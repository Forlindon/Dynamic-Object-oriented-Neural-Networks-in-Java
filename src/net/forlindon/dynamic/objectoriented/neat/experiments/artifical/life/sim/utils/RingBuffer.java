package net.forlindon.dynamic.objectoriented.neat.experiments.artifical.life.sim.utils;

public class RingBuffer<T> {
    private final T[] buffer;
    private final int capacity;
    private int head = 0, tail = 0, count = 0;

    @SuppressWarnings("unchecked")
    public RingBuffer(int capacity) {
        this.capacity = capacity;
        this.buffer = (T[]) new Object[capacity];
    }

    public boolean offer(T item) {
        if (this.count == this.capacity ) return false; // Leave one empty
        this.buffer[this.head] = item;
        this.head = (this.head + 1) % this.capacity;
        this.count++;
        return true;
    }

    public T pop() {
        if (this.count == 0) return null;
        T item = this.buffer[this.tail];
        this.buffer[this.tail] = null;
        this.tail = (this.tail + 1) % this.capacity;
        this.count--;
        return item;
    }

    public double getFillRatio() {
        return (double) this.count / this.capacity;
    }

    public int getCapacity() {
        return this.capacity;
    }

    public int size() {
        return this.getCapacity();
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("RingBuffer[capacity=").append(this.capacity)
                .append(", count=").append(this.count)
                .append(", fillRatio=").append(String.format("%.2f", getFillRatio()))
                .append(", head=").append(this.head)
                .append(", tail=").append(this.tail)
                .append("]\n\tContent: [");
        int pos = this.tail;
        for (int i = 0; i < this.count; i++) {
            sb.append(this.buffer[pos] == null ? "null" : this.buffer[pos].toString());
            if (i < this.count - 1) sb.append(", ");
            pos = (pos + 1) % this.capacity;
        }
        sb.append("]");
        return sb.toString();
    }

    public boolean isEmpty() {
        return this.count == 0;
    }
}
