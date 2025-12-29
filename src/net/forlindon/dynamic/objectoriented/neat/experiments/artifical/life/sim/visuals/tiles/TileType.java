package net.forlindon.dynamic.objectoriented.neat.experiments.artifical.life.sim.visuals.tiles;

public enum TileType {

    GRASS,
    SAND,
    WATER;

    public static int types() {
        return TileType.values().length;
    }


    public static int valueOf(BasicTile basicTile) {
        if (basicTile.getClass() == GrassTile.class) {
            return GRASS.ordinal();
        }
        else if (basicTile.getClass() == SandTile.class) {
            return SAND.ordinal();
        }
        else {
            return WATER.ordinal();
        }
    }
}
