async function fetchStats(i) {
    try {
      const res = await fetch(`/stats${i}`);
      if (!res.ok) return;
      const data = await res.json();
  
      // Time left
      const timeLeft = data && typeof data.time_left !== "undefined" ? data.time_left : "—";
      console.log(timeLeft)
      document.getElementById(`time${i}`).textContent = `Time Left: ${timeLeft}s`;
  
      // Vehicle counts
      const vehicles = data && data.vehicles ? data.vehicles : {};
      let table = document.getElementById(`veh${i}`);
      table.innerHTML = `
        <tr class="bg-slate-800 text-blue-400 text-sm font-semibold">
          <th class="px-2 py-1">Type</th>
          <th class="px-2 py-1">Count</th>
        </tr>`;
      Object.keys(vehicles).forEach((k) => {
        table.innerHTML += `
          <tr class="border-t border-slate-700 text-sm">
            <td class="px-2 py-1 text-center">${k}</td>
            <td class="px-2 py-1 text-center">${vehicles[k]}</td>
          </tr>`;
      });
  
      // Reset lights
      ["red", "yellow", "green"].forEach((c) =>
        document.getElementById(`L${i}${c}`).className =
          "light w-6 h-6 rounded-full bg-slate-700 opacity-30"
      );
  
      // Update current light
      const sig = data && data.signal ? data.signal : "RED";
      if (sig === "RED")
        document.getElementById(`L${i}red`).className =
          "light w-6 h-6 rounded-full bg-red-500 shadow-[0_0_15px_#ef4444]";
      if (sig === "GREEN")
        document.getElementById(`L${i}green`).className =
          "light w-6 h-6 rounded-full bg-green-500 shadow-[0_0_15px_#22c55e]";
      if (sig === "YELLOW")
        document.getElementById(`L${i}yellow`).className =
          "light w-6 h-6 rounded-full bg-yellow-400 shadow-[0_0_15px_#eab308]";
    } catch (err) {
      console.error("Error fetching stats:", err);
    }
  }
  
  // Refresh every 500ms
  setInterval(() => [1, 2, 3].forEach(fetchStats), 500);
  