package net.forlindon.dynamic.objectoriented.neural.networks.layer;

import net.forlindon.dynamic.objectoriented.neural.networks.connection.Connection;
import net.forlindon.dynamic.objectoriented.neural.networks.knot.Knot;
import net.forlindon.dynamic.objectoriented.neural.networks.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.function.BiFunction;
import java.util.function.Function;

public abstract class Layer {

    private final int LAYER_ID;
    protected final List<Knot> KNOTS;

    public Layer(int id) {
        this.LAYER_ID = id;
        this.KNOTS = new ArrayList<>();
    }

    public void add(Function<Integer, Knot> factory) {
        Knot k = factory.apply(this.LAYER_ID);
        this.KNOTS.add(k);
    }

    public abstract void forward();

    public void fullConnect(Layer other, BiFunction<Knot, Knot, Connection> factory) {
        for (Knot a : this.KNOTS) {
            for (Knot b : other.KNOTS) {
                a.connect(b, factory);
            }
        }
    }

    public void readValues(double[] vals) {
        if (vals.length != this.KNOTS.size()) throw new IllegalArgumentException("None matching array length");
        for (int i = 0; i < this.KNOTS.size(); i++) {
            vals[i] = this.KNOTS.get(i).value();
        }
    }

    public void forward(double[] inputs) {
        if (this.KNOTS.size() != inputs.length) throw new IllegalArgumentException("Inputs numbers don't match knots");
        for (int i = 0; i < this.KNOTS.size(); i++) {
            Knot k = this.KNOTS.get(i);
            k.reset();
            k.push(inputs[i]);
            k.pop();
        }
    }

    public void backward(boolean init) {
        if (init) this.KNOTS.forEach(k -> k.pushGrad(1));
        backward();
    }

    public abstract void backward();

    public int id() {
        return this.LAYER_ID;
    }

    @Override
    public String toString() {
        return String.format("{%d: %s}", this.LAYER_ID, this.KNOTS);
    }

    public void adjust() {
        for (Tensor t : getParameters()) {
            t.adjust();
        }
    }

    public List<Knot> getKNOTS() {
        return new ArrayList<>(this.KNOTS);
    }

    public List<Tensor> getParameters() {
        List<Tensor> ts = new ArrayList<>();
        for (Knot k : this.getKNOTS()) {
            ts.add(k.BIAS);
            ts.addAll(k.getConnections());
        }
        return ts;
    }

    public abstract void clean();
}
