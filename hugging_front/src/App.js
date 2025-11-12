import { useState } from 'react'
import { usePred } from './usePred';

export default function App() {

  const [path, setPath] = useState('string')
  const [output, setOutput] = useState('')
  const { getPred } = usePred()

  const onSubmit = async (e) => {
    e.preventDefault()
    const result = await getPred(path)
    setOutput(result[0].generated_text)
  }

  return (
    <div>
      <form onSubmit={onSubmit}>
        <input name="input" placeholder='이미지 경로 입력' onChange={(e)=>{setPath(e.target.value)}}/>
        <button type='submit'>pred</button>
      </form>
      <p>result: {output}</p>
    </div>
  );
}