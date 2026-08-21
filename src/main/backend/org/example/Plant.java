package org.example;

public class Plant {
    public String name;
    public int x, y;
    public int moisture;
    public int lastWatered;
    public int plantType;

    public Plant(String name, int x, int y, int moisture, int lastWatered, int plantType) {
        this.name        = name;
        this.x           = x;
        this.y           = y;
        this.moisture    = moisture;
        this.lastWatered = lastWatered;
        this.plantType   = plantType;
    }
}