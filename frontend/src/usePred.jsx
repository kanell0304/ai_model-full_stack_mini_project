import axios from 'axios'

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8081'

const api = axios.create({
    baseURL: API_BASE_URL,
    headers: {
        'Content-Type': 'application/json',
    },
})

export const usePred = () => {

  const getPred = async (path) => {
    try {
      const response = await api.post('/predict', { path })
      return response.data
    } catch (err) {
      console.error(err.response?.data || err.message)
      throw err 
    }
  }
  return {getPred}
}