# Transcript Style Analyzer

This application analyzes transcript chunks to understand a speaker's unique communication style and generates training data for fine-tuning language models to replicate that style in social media posts.

## Setup

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)

### Installation

1. Clone the repository:
git clone <repository-url>
cd transcript-style-analyzer

2. Create a virtual environment:
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

3. Install dependencies:
pip install -r requirements.txt

4. Create a .env file in the root directory with your xAI API credentials:
XAI_API_KEY=your_api_key_here
XAI_API_URL=https://api.x.ai/v1

## Transcript Preparation

1. Create transcript files in the 'transcripts' directory
2. Format requirements:
   - Files should be in .txt format
   - Each file should contain one transcript chunk
   - Recommended chunk size: 2-5 minutes of speech
   - Example structure:
   
   Speaker: This is the transcript content...
   It can span multiple lines...
   And should capture natural speech patterns.

## Running the Application

1. Ensure your virtual environment is activated
2. Run the main script:
python main.py

## Output

The application generates two files in the 'output' directory:
- training.jsonl: Training examples (80% of data)
- validation.jsonl: Validation examples (20% of data)

Each run clears and regenerates these files to ensure fresh analysis.

## File Structure

transcript-style-analyzer/
├── .env                  # API credentials (create this)
├── .gitignore           # Git ignore rules
├── README.md            # This file
├── requirements.txt     # Python dependencies
├── main.py             # Main application script
├── app.py              # Core application logic
├── transcripts/        # Input transcript files
│   └── .gitkeep
└── output/             # Generated training data
    ├── training.jsonl
    └── validation.jsonl

## Environment Variables

- XAI_API_KEY: Your xAI API key
- XAI_API_URL: xAI API base URL (default: https://api.x.ai/v1)

## Error Handling

The application includes retry logic for API calls and will log:
- Debug information to console
- Errors in case of API failures
- Warnings for retry attempts

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

[Add your license information here]