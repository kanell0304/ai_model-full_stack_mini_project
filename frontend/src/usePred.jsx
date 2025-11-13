export function usePred() {
  const getPred = async (formData) => {
    const res = await fetch('http://localhost:8081/predict', {
      method: 'POST',
      body: formData,
    });

    if (!res.ok) {
      throw new Error(`HTTP error! status: ${res.status}`);
    }

    return res.json();
  };

  return { getPred };
}
