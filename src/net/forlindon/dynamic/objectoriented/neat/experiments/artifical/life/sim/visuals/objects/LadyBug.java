package net.forlindon.dynamic.objectoriented.neat.experiments.artifical.life.sim.visuals.objects;

import net.forlindon.dynamic.objectoriented.neat.experiments.artifical.life.sim.visuals.Vec2d;
import net.forlindon.dynamic.objectoriented.neat.experiments.artifical.life.sim.visuals.tiles.BasicTile;
import net.forlindon.dynamic.objectoriented.neat.experiments.artifical.life.sim.visuals.tiles.TileType;
import net.forlindon.dynamic.objectoriented.neat.experiments.artifical.life.sim.visuals.tiles.WaterTile;
import net.forlindon.dynamic.objectoriented.neat.experiments.artifical.life.sim.visuals.world.World;
import net.forlindon.dynamic.objectoriented.neat.genetic.Genome;
import net.forlindon.dynamic.objectoriented.neat.genetic.PhenoType;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.Comparator;
import java.util.Optional;

public class LadyBug extends Entity {

    public static final int LADY_BUG_SIZE = BasicTile.TILE_SIZE/2;
    public static final BufferedImage LADY_BUG;
    public static final int SIGHT_RANGE = BasicTile.TILE_SIZE*10;
    public static final int SIGHT_DEGREE = 180;
    public static final int N_RAYS = 20;
    public static final int TILE_OBS = 3; // 3x3
    public static final int OBS_SPACE = 2*N_RAYS + TILE_OBS * TILE_OBS + 4; // Ray_Casts(distance, type) + TILE_OBSxTILE_OBS + HEALTH + HUNGER + VelocityX + VelocityY
    public static final int ACT_SPACE = 3;
    public static final int TIME_TILL_ACTION = 3;

    int reproductionTick = 0;
    int actionTick = 0;

    public double[] out = new double[ACT_SPACE];

    public Color color;

    public Genome genome;
    public PhenoType phenoType;

    public static final double HEALTH_REGENERATION_RATE = 0.1;

    static {
        try {
            LADY_BUG = ImageIO.read(new File("resources/entities/ladybug.png"));
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    public LadyBug(World world, int x, int y) {
        super(world, x, y, LADY_BUG_SIZE, LADY_BUG_SIZE);
        set();
        this.world.speciesManager.getGenome(this);
        this.color = new Color((int) (255*Math.random()), (int) (255*Math.random()), (int) (255*Math.random()));
    }

    protected LadyBug (World world, int x, int y, Genome g) {
        super(world, x, y, LADY_BUG_SIZE, LADY_BUG_SIZE);
        set();
        this.hunger=hungerThreshold;
        this.genome = g;
        this.world.speciesManager.getMutationFactory().mutateSelf(g);
        this.phenoType = this.world.speciesManager.phenoTypeBuilder.build(g);
    }

    private void set() {
        this.maxHealth = 20;
        this.health = this.maxHealth;
        this.maxHunger = 320;
        this.hunger = this.maxHunger/2;
        this.hungerThreshold = maxHunger-Bush.HUNGER_PER_BERRY; // 320-55 = 265
        this.reproductionTickThreshold = 40;
        this.reproductionThreshold = maxHunger - Entity.PASSIVE_HUNGER * reproductionTickThreshold * 2; // 310
    }

    @Override
    public void draw(Graphics2D g2d) {
        g2d.setColor(this.color);
        g2d.drawOval(x(),y(),w(),h());
        g2d.setColor(Color.BLACK);
        Object.draw(LADY_BUG, g2d, x(), y(), w(), h(), this.velocity.norm());
        /*
        g2d.drawOval((int) (x()-SIGHT_RANGE/2.0+w()/2.0), (int) (y()-SIGHT_RANGE/2.0+h()/2.0),SIGHT_RANGE,SIGHT_RANGE);
        g2d.setColor(Color.RED);
        for (int i = 0; i < SIGHT_DEGREE; i+=SIGHT_DEGREE / N_RAYS) {
            drawLineFromHead(g2d,i-SIGHT_DEGREE/2.0);
        }
        g2d.setColor(Color.BLACK);
        */
    }

    public void drawLineFromHead(Graphics2D g2d, double degree) {
        Vec2d norm = this.velocity.norm();

        double radX = (w()-w()/4.0)/2.0;
        double radY = (h()-h()/4.0)/2.0;

        Vec2d head = new Vec2d(x()+w()/2.0+norm.getX()*radX,y()+h()/2.0+norm.getY()*radY);

        double dx = SIGHT_RANGE/2.0-radX;
        double dy = SIGHT_RANGE/2.0-radY;
        norm = norm.rotateBy(degree);
        double maxX = head.getX() + norm.getX() * dx;
        double maxY = head.getY() + norm.getY() * dy;

        g2d.drawLine((int) head.getX(), (int) head.getY(), (int) maxX, (int) maxY);
    }

    @Override
    public void tick() {
        super.tick();

        if (++actionTick%TIME_TILL_ACTION==0) {
            double[] observation = getObservation();
            this.phenoType.forward(observation,out);
            actionTick=0;
        }
        if (out[ACT_SPACE-1] >= 0.9) move(new Vec2d(out[0],out[1]));

        if (health <= 0) {
            kill();
            return;
        }

        if (getTile() instanceof WaterTile || hunger <= 0) health--;
        if (hunger >= hungerThreshold) health = Math.min(maxHealth, health + HEALTH_REGENERATION_RATE);

        if (isHungry()) {
            Optional<Object> any = this.world.getObjects(Bush.class).stream().filter(object -> object.start.sub(this.start).length() < BasicTile.TILE_SIZE*2.0/3.0).findAny();
            any.ifPresent(object -> ((Bush) object).eatBerry(this));
            this.reproductionTick = 0;
        }

        if (canReproduce()) this.reproductionTick++;
        else this.reproductionTick = 0;

        if (reproductionTick >= this.reproductionTickThreshold) {
            this.reproductionTick = 0;
            LadyBug ladyBug = new LadyBug(this.world,x(),y(),this.genome.copy());
            ladyBug.color = new Color(shiftRGB(this.color.getRed()), shiftRGB(this.color.getGreen()), shiftRGB(this.color.getBlue()));
            this.hunger = maxHunger/2;
            this.world.OBJECTS_BUFFER.add(ladyBug);
            this.world.speciesManager.add(ladyBug.genome);
        }
    }

    private int shiftRGB(int val) {
        return Math.min(Math.max(0, (int) (val+(Math.random()*2-1))), 255);
    }

    public boolean isHungry() {
        return this.hunger <= hungerThreshold;
    }

    public boolean canReproduce() {
        return this.hunger > this.reproductionThreshold;
    }

    @Override
    public void move(Vec2d vec) {
        super.move(vec);
        if (canMove(vec)) this.hunger-=this.velocity.length()*2./3.;
    }

    public double[] getObservation() {
        double[] inputs = new double[OBS_SPACE];

        int idx = 0;

        inputs[idx++] = this.health;
        inputs[idx++] = this.hunger;
        inputs[idx++] = this.velocity.getX();
        inputs[idx++] = this.velocity.getY();

        Vec2d norm = this.velocity.norm();

        double radX = (w()-w()/4.0)/2.0;
        double radY = (h()-h()/4.0)/2.0;

        Vec2d head = new Vec2d(x()+w()/2.0+norm.getX()*radX,y()+h()/2.0+norm.getY()*radY);
        double r = SIGHT_RANGE/2.0;

        for (int degree = 0; degree < SIGHT_DEGREE; degree+=SIGHT_DEGREE/N_RAYS) {
            Vec2d direction = norm.rotateBy(degree-SIGHT_DEGREE/2.0);
            Optional<Object> first = this.world.OBJECTS.stream().filter(object -> object != this && object.inRay(head, direction, r)).min(Comparator.comparingDouble(o -> head.distance(o.getCenter())));
            if (first.isPresent()) {
                inputs[idx++] = head.distance(first.get().getCenter());
                inputs[idx++] = first.get().encode();
            }
        }

        Vec2d center = this.getCenter();

        for (int i = 0; i < TILE_OBS*TILE_OBS; i++) {
            inputs[idx+i] = -1;
        }

        for (int i = -TILE_OBS/2; i < TILE_OBS/2; i++) {
            for (int j = -TILE_OBS/2; j < TILE_OBS/2; j++) {
                Vec2d pos = center.add(new Vec2d(i*BasicTile.TILE_SIZE,j*BasicTile.TILE_SIZE));
                int x = (int) (pos.getX()/BasicTile.TILE_SIZE);
                int y = (int) pos.getY()/BasicTile.TILE_SIZE;
                if (x < 0 || x > this.world.cols() || y < 0 || y > this.world.rows()) continue;
                inputs[idx++] = TileType.valueOf(this.world.get(x,y));
            }
        }
        return inputs;
    }

    @Override
    public double encode() {
        return 2;
    }

    @Override
    public void kill() {
        this.world.speciesManager.remove(this);
        super.kill();
    }

    @Override
    public boolean canCollide() {
        return false;
    }
}
