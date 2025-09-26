package net.forlindon.dynamic.objectoriented.neural.networks.layer;

import net.forlindon.dynamic.objectoriented.neural.networks.connection.Connection;
import net.forlindon.dynamic.objectoriented.neural.networks.knot.Knot;

import java.util.ArrayList;
import java.util.List;
import java.util.function.BiFunction;
import java.util.function.Function;

public abstract class Layer {

    private final int LAYER_ID;
    protected final List<Knot> PARAMETERS;

    public Layer(int id) {
        this.LAYER_ID = id;
        this.PARAMETERS = new ArrayList<>();
    }

    public void add(Function<Integer, Knot> factory) {
        Knot k = factory.apply(this.LAYER_ID);
        this.PARAMETERS.add(k);
    }

    public abstract void forward();

    public void fullConnect(Layer other, BiFunction<Knot, Knot, Connection> factory) {
        for (Knot a : this.PARAMETERS) {
            for (Knot b : other.PARAMETERS) {
                a.connect(b, factory);
            }
        }
    }

    public void readValues(double[] vals) {
        if (vals.length != this.PARAMETERS.size()) throw new IllegalArgumentException("None matching array length");
        for (int i = 0; i < this.PARAMETERS.size(); i++) {
            vals[i] = this.PARAMETERS.get(i).value();
        }
    }

    public void forward(double[] inputs) {
        if (this.PARAMETERS.size() != inputs.length) throw new IllegalArgumentException("Inputs numbers don't match knots");
        for (int i = 0; i < this.PARAMETERS.size(); i++) {
            Knot k = this.PARAMETERS.get(i);
            k.reset();
            k.push(inputs[i]);
            k.pop();
        }
    }

    public abstract void backward();

    public int id() {
        return this.LAYER_ID;
    }

    public List<Knot> getParameters() {
        return new ArrayList<>(this.PARAMETERS);
    }

    @Override
    public String toString() {
        return String.format("{%d: %s}", this.LAYER_ID, this.PARAMETERS);
    }
}
