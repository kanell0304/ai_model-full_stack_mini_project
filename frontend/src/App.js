import { useState } from 'react'
import { usePred } from './usePred'

export default function App() {
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const [output, setOutput] = useState('')
  const { getPred } = usePred()

  const onSubmit = async (e) => {
    e.preventDefault()
    const result = await getPred(file)
    setOutput(result[0].generated_text)
  }
  
  const fileupload = (e) => {
    const select = e.target.files[0]
    setFile(select)
    setPreview(URL.createObjectURL(select)) // 브라우저 임시 URL, 로컬 아님
  }

  return (
    <div>
      <form onSubmit={onSubmit}>
        <input type="file" accept="image/*"onChange={fileupload}/>
        <button type="submit">예측하기</button>
      </form>
      
      {preview&& <img src={preview} alt="preview" width="300"/>}
      {output&& <p>결과: {output}</p>}
    </div>
  )
}