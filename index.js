import React, { useState } from 'react';

const MyComponent = () => {
  const [prompt, setPrompt] = useState('');
  const [generatedText, setGeneratedText] = useState('');

  const handlePromptChange = (e) => {
    setPrompt(e.target.value);
  };

  const handleGenerateClick = () => {
    fetch('http://192.168.12.105:5000/generate', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ input_text: prompt })
    })
    .then(response => response.json())
    .then(data => {
      const generatedText = data.generated_text;
      setGeneratedText(generatedText);
    })
    .catch(error => {
      console.error('Error: ', error);
    });
  };

  return (
    <div>
      <textarea value={prompt} onChange={handlePromptChange} />
      <button onClick={handleGenerateClick}>Generate Text</button>
      <div>
        <h2>Generated Text:</h2>
        <p>{generatedText}</p>
      </div>
    </div>
  );
};

export default MyComponent;
