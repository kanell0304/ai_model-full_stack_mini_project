import { useState } from 'react'
import { usePred } from './usePred'

export default function App() {
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const [output, setOutput] = useState('')
  const {getPred} = usePred()

  const onSubmit = async(e) => {
    e.preventDefault()
    if(!file) return
    const result = await getPred(file)
    setOutput(result.predicted_class) // FastAPI 반환값 기준
  }

  const fileupload=(e)=>{
    const selected= e.target.files[0]
    if(!selected) return
    setFile(selected)
    setPreview(URL.createObjectURL(selected))
  }

  return (
    <div>
      <form onSubmit={onSubmit}>
        <input type="file" accept="image/*" onChange={fileupload}/>
        <button type="submit">예측하기</button>
      </form>

      {preview && <img src={preview} alt="preview" width="300"/>}
      {output && <p>결과: {output}</p>}
    </div>
  )
}