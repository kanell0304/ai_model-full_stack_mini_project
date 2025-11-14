export function usePred() {
  const getPred = async (formData) => {
    const res = await fetch('http://localhost:8081/predict', {
      method: 'POST',
      body: formData, // FormData 그대로 전송
    });

    if (!res.ok) {
      throw new Error(`HTTP error! status: ${res.status}`);
    }

    return res.json(); // JSON 응답 리턴
  };

  return { getPred };
}
