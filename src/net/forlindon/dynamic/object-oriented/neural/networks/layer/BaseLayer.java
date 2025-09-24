package layer;

import knot.Knot;

public class BaseLayer extends Layer {
    public BaseLayer(int id) {
        super(id);
    }

    @Override
    public void forward() {
        this.parameters.forEach(Knot::pop);
    }
}
