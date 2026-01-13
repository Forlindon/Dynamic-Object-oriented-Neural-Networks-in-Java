package net.forlindon.dynamic.objectoriented.neat.experiments.artifical.life.sim.visuals.world;

import net.forlindon.dynamic.objectoriented.neat.experiments.artifical.life.sim.visuals.tiles.*;

import java.util.ArrayList;
import java.util.List;

public class WorldRule {

    public static void shape(World world) {

        Grid g = world.TILE_MAP.copy();
        for (int y = 0; y < world.rows(); y++) {
            for (int x = 0; x < world.cols(); x++) {
                BasicTile basicTile = world.get(x,y);
                List<Integer> neighbors = getNeighbors(world, x,y);
                int waterN = neighbors.get(TileType.WATER.ordinal());
                int grassN = neighbors.get(TileType.GRASS.ordinal());
                int sandN = neighbors.get(TileType.SAND.ordinal());

                if (basicTile instanceof WaterTile) {
                    if (waterN <= 1) g.set(new GrassTile(x,y));
                    else if (waterN >= 5 || waterN + sandN >= 6) {
                        g.set(new WaterTile(x,y));
                    }
                    else {
                        g.set(new SandTile(x,y));
                    }
                }
                else if (basicTile instanceof GrassTile) {
                    if (grassN >= 3 && waterN == 0) {
                        g.set(new GrassTile(x,y));
                    }
                    else if (waterN >= 1) {
                        g.set(new SandTile(x,y));
                    }
                    else {
                        g.set(new GrassTile(x,y));
                    }
                }
                else if (basicTile instanceof SandTile) {
                    if (waterN == 0) {
                        g.set(new GrassTile(x,y));
                    }
                    else if (waterN >= 5 || grassN == 0 && waterN >= 3) {
                        g.set(new WaterTile(x,y));
                    }
                    else {
                        g.set(new SandTile(x,y));
                    }
                }
            }
        }
        world.TILE_MAP.fill(g);
    }

    public static List<Integer> getNeighbors(World world, int x, int y) {
        int xStart = Math.max(x-1,0);
        int xEnd = Math.min(x+1,world.cols()-1);

        int yStart = Math.max(y-1,0);
        int yEnd = Math.min(y+1,world.rows()-1);

        List<Integer> integers = new ArrayList<>(TileType.types());
        for (int i = 0; i < TileType.types(); i++) {
            integers.add(0);
        }

        for (int i = yStart; i <= yEnd; i++) {
            for (int j = xStart; j <= xEnd; j++) {
                BasicTile basicTile = world.get(j,i);
                if (i == y && j == x) continue;
                int idx = TileType.valueOf(basicTile);
                integers.set(idx,integers.get(idx)+1);
            }
        }

        return integers;
    }

    public static void cleanUp(World world) {
        Grid grid = world.TILE_MAP;
        for (int y = 0; y < world.rows(); y++) {
            for (int x = 0; x < world.cols(); x++) {
                BasicTile basicTile = world.get(x,y);
                List<Integer> neighbors = getNeighbors(world, x, y);
                int waterN = neighbors.get(TileType.WATER.ordinal());
                int grassN = neighbors.get(TileType.GRASS.ordinal());

                if (basicTile instanceof GrassTile && waterN >= 1) {
                    grid.set(new SandTile(x,y));
                }
                else if (basicTile instanceof SandTile) {
                    if (waterN < 1) grid.set(new GrassTile(x,y));
                    else if (grassN == 0) grid.set(new WaterTile(x,y));
                }
            }
        }
        world.TILE_MAP.fill(grid);
    }
}
