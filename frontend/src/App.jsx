import { useState } from 'react'
import { usePred } from './usePred'
import './App.css'

export default function App() {
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const [output, setOutput] = useState('')
  const {getPred} = usePred()

  const onSubmit = async(e) => {
    e.preventDefault()
    setOutput({"predicted_class": "Loading...", "confidence": "Loading..."})
    if(!file) return
    const result = await getPred(file)
    setOutput(result) // FastAPI 반환값 기준
  }

  const fileupload=(e)=>{
    const selected= e.target.files[0]
    if(!selected) return
    setFile(selected)
    setPreview(URL.createObjectURL(selected))
  }

  return (
    <div className="container">
      <h2>이미지 분류기</h2>
      <form onSubmit={onSubmit} className="upload-box">
        <input type="file" accept="image/*" onChange={fileupload}/>
        <button type="submit">예측하기</button>
      </form>

      {preview && (<div className="preview"><img src={preview} alt="preview"/></div>)}

      {output && (
        <div className="result-box">
          <p><strong>결과</strong>: {output.predicted_class}</p>
          <p><strong>정확도</strong>: {output.confidence}</p>
        </div>
      )}
    </div>
  )
}