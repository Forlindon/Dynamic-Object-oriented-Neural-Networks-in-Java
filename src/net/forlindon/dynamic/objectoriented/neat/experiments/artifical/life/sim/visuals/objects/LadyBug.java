package net.forlindon.dynamic.objectoriented.neat.experiments.artifical.life.sim.visuals.objects;

import net.forlindon.dynamic.objectoriented.neat.experiments.artifical.life.sim.neat.BufferEntry;
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
import java.util.Comparator;
import java.util.List;
import java.util.Optional;
import java.util.stream.Stream;

public class LadyBug extends Entity {

    public static final int LADY_BUG_SIZE = BasicTile.TILE_SIZE/2;
    public static final double LADY_BUG_STEP = LADY_BUG_SIZE/8.;
    public static final BufferedImage LADY_BUG;
    public static final int SIGHT_RANGE = BasicTile.TILE_SIZE*6;
    public static final int SIGHT_DEGREE = 90;
    public static final int N_RAYS = 16;
    public static final int TILE_OBS = 3; // 3x3
    public static final int OBS_SPACE = 2*N_RAYS + TILE_OBS * TILE_OBS + 6; // Ray_Casts(distance, type) + TILE_OBSxTILE_OBS + HEALTH + HUNGER + VelocityX + VelocityY + Age + ReproductionTick
    public static final int ACT_SPACE = 4;
    public static final int TIME_TILL_ACTION = 7;
    public static final int MAX_AGE = 2*1200; // n * ticks per minute => age
    public static final int COLOR_SHIFT = 13;
    public static final int FITNESS_PER_BIRTH = 10;
    private static final double HUNGER_PER_BIRTH = Bush.HUNGER_PER_BERRY;

    int reproductionTick = 0;
    int actionTick = 0;

    public double[] in = new double[OBS_SPACE];
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
        this.genome = g;
        this.world.speciesManager.getMutationFactory().mutateSelf(g);
        this.phenoType = this.world.speciesManager.phenoTypeBuilder.build(g);
    }

    public LadyBug(LadyBug a, LadyBug b) {
        this(a.world, (a.x()+b.x())/2,(a.y()+b.y())/2, Genome.crossOver(a.genome,b.genome));
        this.color = new Color((a.color.getRed()+b.color.getRed())/2,(a.color.getGreen()+b.color.getGreen())/2,(a.color.getBlue()+b.color.getBlue())/2);
        b.reproductionTick = 0;
        b.hunger += HUNGER_PER_BIRTH;
        b.genome.setFitness(b.genome.getFitness()+FITNESS_PER_BIRTH);
        this.genome.setFitness(0);
    }

    private void set() {
        this.maxHealth = 20;
        this.health = this.maxHealth;
        this.maxHunger = 320;
        this.hunger = maxHunger/2;
        this.hungerThreshold = Bush.HUNGER_PER_BERRY;
        this.reproductionTickThreshold = 20;
        this.maxAge = MAX_AGE;
        this.age = 0;
        this.reproductionThreshold = maxHunger/2.;
    }

    @Override
    public void draw(Graphics2D g2d) {
        g2d.setColor(this.color);
        g2d.drawOval(x(),y(),w(),h());
        g2d.setColor(Color.BLACK);
        Object.draw(LADY_BUG, g2d, x(), y(), w(), h(), this.velocity.norm());
        /*
        g2d.setColor(Color.RED);
        for (double i = 0.; i < SIGHT_DEGREE; i+= (double) SIGHT_DEGREE / N_RAYS) {
            // drawLineFromHead(g2d,i-SIGHT_DEGREE/2.0);
            drawLineFromCenter(g2d,i-SIGHT_DEGREE/2.0);
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

    public void drawLineFromCenter(Graphics2D g2d, double degree) {
        Vec2d norm = this.velocity.norm();

        Vec2d cent = getCenter();

        double r = SIGHT_RANGE/2.;
        norm = norm.rotateBy(degree);
        if (Double.isNaN(norm.getX()) || Double.isNaN(norm.getY())) {
            norm = new Vec2d(0,1).rotateBy(degree);
        }
        double maxX = cent.getX() + norm.getX() * r;
        double maxY = cent.getY() + norm.getY() * r;

        Vec2d direction = norm;
        List<Object> objects = this.world.OBJECTS.stream().filter(object -> object != this && object.inRay(cent, direction, r)).toList();
        objects.forEach(object -> g2d.drawRect(object.x(),object.y(),object.w(), object.h()));
        g2d.drawLine((int) cent.getX(), (int) cent.getY(), (int) maxX, (int) maxY);
    }

    @Override
    public void tick() {
        super.tick();

        if (++actionTick%TIME_TILL_ACTION==0) {
            takeAction();
        }
        if (wantsToMove()) move(new Vec2d(out[0]*LADY_BUG_STEP,out[1]*LADY_BUG_STEP));

        if (getTile() instanceof WaterTile || hunger >= maxHunger) health--;
        if (!isHungry()) health = Math.min(maxHealth, health + HEALTH_REGENERATION_RATE);

        if (isHungry()) {
            Optional<Object> any = this.world.getObjects(Bush.class).stream().filter(object -> object.getCenter().distance(getCenter()) <= Bush.BUSH_SIZE/2.+LADY_BUG_SIZE/2. && ((Bush) object).capacity > 0).findAny();
            any.ifPresent(object -> {
                ((Bush) object).eatBerry(this);
            });
            this.reproductionTick = 0;
        }

        if (canReproduce()) this.reproductionTick++;
        else this.reproductionTick = 0;

        if (this.reproductionTick >= this.reproductionTickThreshold) {
            reproduce();
        }
    }

    private void takeAction() {
        double[] observation = getObservation();
        boolean pushed = this.world.getBUFFER().offer(new BufferEntry(this.phenoType, observation, this.out));
        this.actionTick=0;
    }

    public boolean wantsToMove() {
        return this.out[ACT_SPACE-2] > 0.5;
    }

    public boolean wantsToReproduce() {
        return this.out[ACT_SPACE-1] > 0.5;
    }

    public void reproduce() {
        Optional<LadyBug> collidesWith = this.world.getObjects(LadyBug.class).stream().map(object -> (LadyBug) object).filter(object -> this != object && object.canReproduce() && object.intersect(this)).findFirst();
        LadyBug ladyBug;
        if (collidesWith.isPresent()) {
            ladyBug = new LadyBug(this, collidesWith.get());
        }
        else {
            ladyBug = new LadyBug(this.world,x(),y(),this.genome.copy());
            ladyBug.color = new Color(shiftRGB(this.color.getRed()), shiftRGB(this.color.getGreen()), shiftRGB(this.color.getBlue()));
        }
        giveBirth(ladyBug);
    }

    public void giveBirth(LadyBug ladyBug) {
        this.reproductionTick = 0;
        this.hunger += HUNGER_PER_BIRTH;
        this.world.OBJECTS_BUFFER.add(ladyBug);
        this.world.speciesManager.add(ladyBug.genome);
        this.genome.setFitness(genome.getFitness()+FITNESS_PER_BIRTH);
    }

    private static int shiftRGB(int val) {
        return Math.min(Math.max(0, (int) (val+(Math.random()*COLOR_SHIFT-COLOR_SHIFT/2.))), 255);
    }

    public boolean isHungry() {
        return this.hunger >= hungerThreshold;
    }

    public boolean canReproduce() {
        return this.hunger < this.reproductionThreshold && wantsToReproduce();
    }

    @Override
    public void move(Vec2d vec) {
        super.move(vec);
        if (canMove(vec)) {
            this.hunger+=this.velocity.length()*2./3.;
        }
    }

    public double[] getObservation() {

        int idx = 0;

        this.in[idx++] = this.health / this.maxHealth;
        this.in[idx++] = this.hunger / this.maxHunger;
        this.in[idx++] = (double) this.age / MAX_AGE;
        this.in[idx++] = (double) this.reproductionTick / this.reproductionTickThreshold;
        this.in[idx++] = this.velocity.getX();
        this.in[idx++] = this.velocity.getY();

        Vec2d norm = this.velocity.norm();

        Vec2d center = getCenter();

        double r = SIGHT_RANGE/2.0;

        for (double degree = 0; degree < SIGHT_DEGREE; degree+= (double) SIGHT_DEGREE / N_RAYS) {
            Vec2d direction = norm.rotateBy(degree-SIGHT_DEGREE/2.0);
            Optional<Object> first = this.world.OBJECTS.stream().filter(object -> object != this && object.inRay(center, direction, r)).min(Comparator.comparingDouble(o -> center.distance(o.getCenter())));
            if (first.isPresent()) {
                this.in[idx++] = r - center.distance(first.get().getCenter());
                this.in[idx++] = first.get().encode();
            }
        }

        for (int i = 0; i < TILE_OBS*TILE_OBS; i++) {
            this.in[idx+i] = -1;
        }

        for (int i = -TILE_OBS/2; i < TILE_OBS/2; i++) {
            for (int j = -TILE_OBS/2; j < TILE_OBS/2; j++) {
                Vec2d pos = center.add(new Vec2d(i*BasicTile.TILE_SIZE,j*BasicTile.TILE_SIZE));
                int x = (int) (pos.getX()/BasicTile.TILE_SIZE);
                int y = (int) pos.getY()/BasicTile.TILE_SIZE;
                if (x < 0 || x > this.world.cols() || y < 0 || y > this.world.rows()) continue;
                this.in[idx++] = TileType.valueOf(this.world.get(x,y));
            }
        }
        return this.in;
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

    public Vec2d getHead() {
        Vec2d norm = this.velocity.norm();

        double radX = (w()-w()/4.0)/2.0;
        double radY = (h()-h()/4.0)/2.0;

        return new Vec2d(x()+w()/2.0+norm.getX()*radX,y()+h()/2.0+norm.getY()*radY);
    }
}
