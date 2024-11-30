import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any
from dotenv import load_dotenv
import requests
from tqdm import tqdm
import random
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TranscriptProcessor:
    def __init__(self):
        load_dotenv()
        self.xai_api_key = os.getenv('XAI_API_KEY')
        self.xai_api_url = os.getenv('XAI_API_URL')
        self.transcripts_dir = Path('transcripts')
        self.output_dir = Path('output')
        self.output_dir.mkdir(exist_ok=True)
        
        # API rate limiting parameters
        self.max_retries = 3
        self.retry_delay = 5  # seconds
        
    def load_transcripts(self) -> List[Dict[str, Any]]:
        """Load and process all transcript files from the transcripts directory."""
        transcripts = []
        
        try:
            transcript_files = list(self.transcripts_dir.glob('*.txt'))
            if not transcript_files:
                raise FileNotFoundError("No transcript files found in the transcripts directory")
            
            for file_path in tqdm(transcript_files, desc="Loading transcripts"):
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read().strip()
                    transcripts.append({
                        'filename': file_path.name,
                        'content': content
                    })
                    
            logger.info(f"Successfully loaded {len(transcripts)} transcript files")
            return transcripts
            
        except Exception as e:
            logger.error(f"Error loading transcripts: {str(e)}")
            raise
            
    def analyze_style(self, text: str) -> Dict[str, Any]:
        """Analyze text style using xAI API with comprehensive style analysis."""
        headers = {
            'Authorization': f'Bearer {self.xai_api_key}',
            'Content-Type': 'application/json'
        }
        
        # Prepare the messages for detailed style analysis
        payload = {
            "model": "grok-beta",
            "messages": [
                {
                    "role": "system",
                    "content": """Analyze the following transcript excerpt for the speaker's unique communication style. 
                    Focus on:
                    1. Syntactical patterns (sentence structure, length, transitions)
                    2. Vocabulary choices (technical terms, common phrases, industry terms)
                    3. Tone markers (formality, humor, emotional expression)
                    4. Engagement patterns (audience interaction, storytelling)
                    5. Unique characteristics (signature phrases, explanation style)
                    6. Grammar preferences (active/passive voice, contractions)
                    7. Content structure (topic introduction, examples, conclusions)
                    
                    Then, based on this analysis, generate a social media post that authentically replicates 
                    the speaker's voice and style while discussing the requested topic."""
                },
                {
                    "role": "user",
                    "content": f"Transcript to analyze:\n{text}"
                }
            ],
            "temperature": 0.7  # Allow some creativity while maintaining style consistency
        }
        
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    f"{self.xai_api_url}/chat/completions",
                    headers=headers,
                    json=payload
                )
                
                logger.debug(f"Response status: {response.status_code}")
                logger.debug(f"Response content: {response.text}")
                
                response.raise_for_status()
                return response.json()
                
            except requests.exceptions.RequestException as e:
                if attempt == self.max_retries - 1:
                    logger.error(f"Failed to analyze style after {self.max_retries} attempts: {str(e)}")
                    raise
                logger.warning(f"Retry {attempt + 1}/{self.max_retries} after error: {str(e)}")
                time.sleep(self.retry_delay)

    def generate_training_examples(self, style_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate training examples based on the style analysis."""
        # Topics for post generation
        topics = [
            "industry insights", "professional growth", "innovation", "leadership",
            "technology trends", "workplace culture", "success stories", "team building",
            "market analysis", "future predictions", "personal development", 
            "problem solving", "change management", "strategic thinking"
        ]
        
        examples = []
        for topic in random.sample(topics, 5):  # Generate 5 examples per analysis
            example = {
                "messages": [
                    {
                        "role": "system",
                        "content": "You are an assistant that writes social media posts matching the speaker's authentic voice and style."
                    },
                    {
                        "role": "user",
                        "content": f"Write a social media post about {topic}"
                    },
                    {
                        "role": "assistant",
                        "content": self.generate_post_from_style(style_analysis, topic)
                    }
                ]
            }
            examples.append(example)
        
        return examples

    def generate_post_from_style(self, style_analysis: Dict[str, Any], topic: str) -> str:
        """Generate a post based on the analyzed style and requested topic."""
        headers = {
            'Authorization': f'Bearer {self.xai_api_key}',
            'Content-Type': 'application/json'
        }
        
        payload = {
            "model": "grok-beta",
            "messages": [
                {
                    "role": "system",
                    "content": f"""Based on the speaker's analyzed style, generate an authentic social media post about {topic}.
                    Maintain their unique voice characteristics, vocabulary preferences, and structural patterns."""
                },
                {
                    "role": "user",
                    "content": f"Style analysis: {json.dumps(style_analysis)}\nTopic: {topic}"
                }
            ],
            "temperature": 0.7
        }
        
        response = requests.post(
            f"{self.xai_api_url}/chat/completions",
            headers=headers,
            json=payload
        )
        response.raise_for_status()
        
        return response.json()['choices'][0]['message']['content']

    def create_datasets(self, examples: List[Dict[str, Any]]):
        """Split and save training and validation datasets."""
        random.shuffle(examples)
        split_index = int(len(examples) * 0.8)
        
        training_data = examples[:split_index]
        validation_data = examples[split_index:]
        
        self._save_jsonl(training_data, self.output_dir / 'training.jsonl')
        self._save_jsonl(validation_data, self.output_dir / 'validation.jsonl')
        
        logger.info(f"Created datasets: {len(training_data)} training examples, "
                   f"{len(validation_data)} validation examples")

    def _save_jsonl(self, data: List[Dict[str, Any]], filepath: Path):
        """Save data in JSONL format with validation."""
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                for item in data:
                    # Validate required fields
                    if not self._validate_example(item):
                        logger.warning(f"Skipping invalid example: {item}")
                        continue
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
                    
        except Exception as e:
            logger.error(f"Error saving JSONL file {filepath}: {str(e)}")
            raise

    def _validate_example(self, example: Dict[str, Any]) -> bool:
        """Validate the structure of a training example."""
        try:
            messages = example.get('messages', [])
            if len(messages) != 3:
                return False
                
            required_roles = ['system', 'user', 'assistant']
            for msg, role in zip(messages, required_roles):
                if msg.get('role') != role or 'content' not in msg:
                    return False
                    
            return True
            
        except Exception as e:
            logger.error(f"Validation error: {str(e)}")
            return False

    def process(self):
        """Main processing pipeline."""
        try:
            logger.info("Starting transcript processing")
            
            # Load transcripts
            transcripts = self.load_transcripts()
            
            # Analyze style for each transcript
            all_examples = []
            for transcript in tqdm(transcripts, desc="Processing transcripts"):
                style_analysis = self.analyze_style(transcript['content'])
                examples = self.generate_training_examples(style_analysis)
                all_examples.extend(examples)
            
            # Create and save datasets
            self.create_datasets(all_examples)
            
            logger.info("Processing completed successfully")
            
        except Exception as e:
            logger.error(f"Processing failed: {str(e)}")
            raise

def main():
    processor = TranscriptProcessor()
    processor.process()

if __name__ == "__main__":
    main() 