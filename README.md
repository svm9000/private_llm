# Steps to run Retrieval-Augmented Generation (RAG) with Mistral 
This example provides an interface for asking questions to a PDF document with a front end Gradio app.

# 1. Install Ollama using Docker

Ollama can run with GPU acceleration inside Docker containers for Nvidia GPUs.

To get started using the Docker image, please use the commands below.
CPU only

``` 
docker run -d -v /home/test/ollama:/root/ollama -p 11434:11434 --name ollama ollama/ollama
```
Nvidia GPU

- Install the Nvidia container toolkit.
- Run Ollama inside a Docker container

```
docker run -d --gpus=all -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama
```

## install python and required packages inside container
```
docker exec -it ollama /bin/bash
apt-get update
apt-get install -y python3-pip
```

```
pip install -r requirements.txt
```

## Run

```
python main.py
```

A prompt will appear, where questions may be asked:

```
Query: How many locations does WeWork have?
```


### Setup

Set up a virtual environment (optional):

```
python3 -m venv .venv
source .venv/bin/activate
```

Install the Python dependencies:

```shell
pip install -r requirements.txt
```kedi

Pull the model you'd like to use:

```
ollama pull llama2-uncensored
```

### Getting WeWork's latest quarterly earnings report (10-Q)

```
mkdir source_documents
curl https://d18rn0p25nwr6d.cloudfront.net/CIK-0001813756/975b3e9b-268e-4798-a9e4-2a9a7c92dc10.pdf -o source_documents/wework.pdf
```

### Ingesting files

```shell
python ingest.py
```

Output should look like this:

```shell
Creating new vectorstore
Loading documents from source_documents
Loading new documents: 100%|██████████████████████| 1/1 [00:01<00:00,  1.73s/it]
Loaded 1 new documents from source_documents
Split into 90 chunks of text (max. 500 tokens each)
Creating embeddings. May take some minutes...
Using embedded DuckDB with persistence: data will be stored in: db
Ingestion complete! You can now run privateGPT.py to query your documents
```

### Ask questions

```shell
python privateGPT.py

Enter a query: How many locations does WeWork have?

> Answer (took 17.7 s.):
As of June 2023, WeWork has 777 locations worldwide, including 610 Consolidated Locations (as defined in the section entitled Key Performance Indicators).
```

### Try a different model:

```
ollama pull llama2:13b
MODEL=llama2:13b python privateGPT.py
```

## Adding more files

Put any and all your files into the `source_documents` directory

The supported extensions are:

- `.csv`: CSV,
- `.docx`: Word Document,
- `.doc`: Word Document,
- `.enex`: EverNote,
- `.eml`: Email,
- `.epub`: EPub,
- `.html`: HTML File,
- `.md`: Markdown,
- `.msg`: Outlook Message,
- `.odt`: Open Document Text,
- `.pdf`: Portable Document Format (PDF),
- `.pptx` : PowerPoint Document,
- `.ppt` : PowerPoint Document,
- `.txt`: Text file (UTF-8),
