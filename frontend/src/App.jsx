import { useState } from 'react'
import { usePred } from './usePred';
import './App.css'

export default function App() {
  const [file, setFile] = useState(null);
  const [output, setOutput] = useState('');
  const [image, setImage] = useState(null);
  const { getPred } = usePred();

  const onSubmit = async (e) => {
    e.preventDefault();
    if (!file) {
      alert('이미지를 선택하세요');
      return;
    }

    const formData = new FormData();
    formData.append('file', file);

    
    const result = await getPred(formData);
    const percent = (result.confidence * 100).toFixed(2);

    setOutput(`이 사진은 무엇일까요? : ${result.label_name}, 절 이정도 믿으세요 : ${percent}%`);
  };

  const onFileChange=(e)=>{
    const selected = e.target.files && e.target.files[0];

    if(!selected) {
      setFile(null);
      setImage(null);
      return;
    }
    setFile(selected)
    setImage(URL.createObjectURL(selected));
  }

  return (
    <div className='container'>
      <h2>이미지 분류기</h2>
      <form onSubmit={onSubmit} className="upload-box">
        <input type="file" accept="image/*" onChange={onFileChange}/>
        <button type="submit">예측</button>
      </form>
		
			{image && (
			  <div className="preview">
			    <img src={image} alt='업로드한 이미지' />
			  </div>
			)}
			
			<p className="result">{output}</p>
		            
    </div>
  );
}