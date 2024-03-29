## README for Robot Chess Image classifier documentation

### Workflow:

1) Input: Image link Download from Lab Livestream
    1) Alternatively will pull Image from web with requests
2) Slicing Image into individual fields
3) Classification of Images using trained Image calssifier
4) Output is classified 8x8 Matrix as output is sent as response

### Export environment:

```bash
conda env export | grep -v "^prefix: " > environment.yml
```

### Import environment:

```bash
conda env create -f environment.yml
```

### Logging: 

#### Root level logging levels:
```
logger.info()
logger.warning()
logger.error()
```

#### App level logging:
```
EchoService.echo()
```

#### Endpoints:
```
@app.get("/empty_full")
@app.get("/counter")
@app.get("/next_move_to_free_space")
@app.get("/save_samples")
@app.get("/")
```