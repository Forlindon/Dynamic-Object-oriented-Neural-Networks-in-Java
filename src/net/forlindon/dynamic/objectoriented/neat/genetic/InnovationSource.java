package net.forlindon.dynamic.objectoriented.neat.genetic;

import java.util.TreeSet;

public class InnovationSource {

    private int range;

    public InnovationSource() {
        this.range = Integer.MIN_VALUE;
    }

    @Override
    public String toString() {
        return String.valueOf(this.range-Integer.MIN_VALUE);
    }

    public int getNext() {
        return this.range++;
    }

    public boolean isInRange(int i) {
        return i < this.range;
    }
}
