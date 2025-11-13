import axios from 'axios'

const API_BASE_URL = 'http://localhost:8081'

const api = axios.create({
    baseURL: API_BASE_URL,
    headers: {
        'Content-Type': 'application/json',
    },
})

export const usePred = () => {
  const getPred = async(file)=>{
    try{
      const formData = new FormData()
      formData.append('file', file)

      const response = await api.post('/predict', formData, {
        headers: {'Content-Type': 'multipart/form-data'},
      })

      return response.data
    }
    catch(err){
      console.error(err.response?.data || err.message)
      throw err
    }
  }
  return{getPred}
}