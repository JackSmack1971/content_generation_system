# Content Generation System

This project is an advanced content generation system that utilizes modern AI techniques, featuring a plugin-based architecture for customizable content processing. It also includes an intuitive web interface built with **Gradio** for seamless user interaction.

## Features

- **Dynamic Content Generation**: Generate content for multiple topics simultaneously.
- **Multiple Output Formats**: Supports plain text, HTML, Markdown, and PDF.
- **Plugin System**: Extensible plugin architecture to add custom processing steps such as:
  - Spell-checking
  - Grammar correction
  - SEO analysis
- **Feedback Loop**: Iteratively improves content based on user feedback.
- **Progress Tracking**: Real-time feedback and progress tracking for each content generation step.
- **Rate Limiting**: Ensures compliance with API rate limits when utilizing external services.

## Installation

Follow these steps to install and run the project:

1. **Clone the repository:**
    ```bash
    git clone https://github.com/JackSmack1971/content_generation_system.git
    cd content_generation_system
    ```

2. **Set up a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3. **Install project dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4. **Install system dependencies for PDF generation:**
    ```bash
    sudo apt-get install wkhtmltopdf
    ```

5. **Configure environment variables:**
    - Add your API keys and configuration details in `config/settings.py`.

## Usage

To launch the Gradio web interface, run:

```bash
python src/main.py
```

Once the interface is running, you can:
- Input multiple topics for content generation.
- Choose the desired output format: plain text, HTML, Markdown, or PDF.
- Track the generation progress in real-time.
- Submit feedback on the generated content and receive improved versions.

## Folder Structure

```
/content_generation_system
│
├── /src                    # Core logic and main files
│   ├── main.py             # Main entry point for Gradio interface
│   ├── content_processor.py # Core logic for processing content
│   ├── formatters.py       # HTML, Markdown, and PDF formatters
│   └── ...                 # Additional core components
│
├── /plugins                # Custom plugins for content processing
│   └── ...                 # Plugins like SpellCheckPlugin, GrammarCorrectionPlugin
│
├── /tests                  # Unit tests for the project
│   └── ...                 # Tests for various components of the system
│
├── /config                 # Configuration files and environment settings
│   └── settings.py         # API keys, rate limits, and other configuration settings
│
├── /assets                 # Generated assets, such as PDF files
│   └── ...
│
├── requirements.txt        # List of dependencies
├── README.md               # Project documentation
├── .gitignore              # Files and folders to ignore in version control
└── LICENSE                 # Licensing information
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
