# Contributing to RAG

## Project Structure

This repository follows a structured approach to organize demonstrations, benchmarks, and deployment configurations:

```
rag/
├── demos/           # Demonstrations
├── benchmarks/      # Performance benchmarking scripts
├── stack/           # Kubernetes/OpenShift deployment configurations
├── DEPLOYMENT.md    # RAG Stack deployment guide
└── README.md        # Project overview
```

## Contributing Demos

### Demo Structure Requirements

Each demo **must** include the following documentation:

1. **README.md** - Describes the demo's purpose, overview, and usage
2. **DEPLOYMENT.md** (or deployment section in README.md) - Step-by-step reproduction instructions for the demo
3. **requirements.txt** - if necessary

### Demo Folder Organization

Follow this naming pattern for new demos:

```
demos/
├── <platform>/
│   └── <use-case>/
│       ├── README.md
│       ├── DEPLOYMENT.md (or deployment instructions in README.md)
│       ├── requirements.txt (if applicable)
│       └── <demo-specific-files>
```

<!--- TODO: Add more demos to the examples as they are added to the project -->
**Examples:**
- [demos/kfp/pdf-conversion](demos/kfp/pdf-conversion)

### Demo Documentation Standards

Your demo's **README.md** should include:

1. **Title and Overview** - What does this demo do?
2. **Prerequisites** - Required software, resources, or setup
3. **Resource Requirements** - CPU, memory, GPU specifications
4. **Step-by-step Instructions** - Clear deployment/running instructions
5. **Key Parameters** - Configurable options and their purposes
6. **Usage Examples** - How to interact with the deployed demo

### Adding a New Demo

1. **Create the folder structure** following the pattern above
2. **Implement your demo** with proper error handling and documentation
3. **Write comprehensive README.md** following our standards
4. **Test deployment instructions** on a clean environment
5. **Submit a pull request** with your contribution

## Contributing Benchmarks

### Benchmark Structure

Benchmarks should be organized as:

```
benchmarks/
├── <benchmark-name>/
│   ├── README.md
│   ├── <benchmark-script>.py
│   ├── requirements.txt
│   └── <deployment-specific-files> (if applicable)
```

### Benchmark Documentation

Each benchmark should include:

1. **Purpose** - What aspect of performance is being measured
2. **Setup Instructions** - How to prepare the benchmarking environment
3. **Running Instructions** - Command-line usage and parameters
4. **Results Interpretation** - How to understand the output

## Deployment Configurations

When contributing to the `stack/` directory:

- **Follow Kubernetes best practices**
- **Document any new overlays** or configurations in the [DEPLOYMENT.md](DEPLOYMENT.md) guide
- **Test deployments** on both OpenShift or Kubernetes when possible
- **Include resource requirements** in documentation

### pre-commit

This project is configured to use pre-commit for every new PR.
You can find instructions for installing pre-commit [here](https://pre-commit.com/#installation)

## Setup pre-commit for the RAG project

Run the following command to allow pre-commit to run before each commit:

``` bash
pre-commit install
```

To run pre-commit without commiting run:

``` bash
pre-commit run --all-files
```
