package layer;

import knot.Knot;

import java.util.ArrayList;
import java.util.List;

public abstract class Layer {

    private final int LAYER_ID;
    List<Knot> parameters;

    public Layer(int id) {
        this.LAYER_ID = id;
        this.parameters = new ArrayList<>();
    }

    public void add(Knot k) {
        if (k.id() <= this.LAYER_ID) throw new IllegalArgumentException("Illegal Layer ID");
        this.parameters.add(k);
    }

}
