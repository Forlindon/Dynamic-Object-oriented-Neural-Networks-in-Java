package layer;

import connection.Connection;
import knot.Knot;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Function;

public abstract class Layer {

    private final int LAYER_ID;
    List<Knot> parameters;

    public Layer(int id) {
        this.LAYER_ID = id;
        this.parameters = new ArrayList<>();
    }

    public void add(Function<Integer, Knot> factory) {
        Knot k = factory.apply(this.LAYER_ID);
        this.parameters.add(k);
    }

    public abstract void forward();

    public void fullConnect(Layer other, Function<Knot, Connection> factory) {
        for (Knot a : this.parameters) {
            for (Knot b : other.parameters) {
                a.connect(b, factory);
            }
        }
    }

    public void readValues(double[] vals) {
        if (vals.length != this.parameters.size()) throw new IllegalArgumentException("None matching array length");
        for (int i = 0; i < this.parameters.size(); i++) {
            vals[i] = this.parameters.get(i).value();
        }
    }

    public void forward(double[] inputs) {
        if (this.parameters.size() != inputs.length) throw new IllegalArgumentException("Inputs numbers don't match knots");
        for (int i = 0; i < this.parameters.size(); i++) {
            Knot k = this.parameters.get(i);
            k.reset();
            k.push(inputs[i]);
            k.pop();
        }
    }

    @Override
    public String toString() {
        return String.format("{%d: %s}", this.LAYER_ID, this.parameters);
    }
}
