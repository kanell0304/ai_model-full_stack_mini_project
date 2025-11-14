import { useState } from 'react'
import { usePred } from './usePred';
import './App.css'

export default function App() {
  const [file, setFile] = useState(null); // 실제 파일 데이터
  const [output, setOutput] = useState(''); // 결과 문구
  const [image, setImage] = useState(null); // 미리보기 이미지 url
  const { getPred } = usePred();

  // 폼 제출
  const onSubmit = async (e) => {
    e.preventDefault();
    if (!file) {
      alert('이미지를 선택하세요');
      return;
    }

    const formData = new FormData();
    formData.append('file', file); // FormData에 file이라는 이름으로 이미지 파일을 담음

    
    const result = await getPred(formData); // getPred 호출
    const percent = (result.confidence * 100).toFixed(2);

    setOutput(`이 사진은 무엇일까요? : ${result.label_name}, 절 이정도 믿으세요 : ${percent}%`);
  };

  // 파일 선택
  const onFileChange=(e)=>{
    const selected = e.target.files && e.target.files[0];

    if(!selected) { // 선택된 사진이 없으면 null
      setFile(null);
      setImage(null);
      return;
    }

    setFile(selected) // 있으면 selected 실제 파일 저장
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