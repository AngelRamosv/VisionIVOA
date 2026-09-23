import re

with open("src/cloud_backend/templates/dashboard.html", "r", encoding="utf-8") as f:
    code = f.read()

old_chart_logic = '''            const ctx = document.getElementById('incidenciasChart').getContext('2d');
            if (incidenciasChart) {
                incidenciasChart.data.labels = activeLabels;
                incidenciasChart.data.datasets[0].data = activeValues;
                incidenciasChart.data.datasets[0].backgroundColor = activeColors;
                incidenciasChart.data.datasets[0].borderColor = activeColors;
                incidenciasChart.update();
            } else {
                incidenciasChart = new Chart(ctx, {
                    type: 'bar',
                    data: {
                        labels: activeLabels,
                        datasets: [{
                            label: 'Tickets Emitidos',
                            data: activeValues,
                            backgroundColor: activeColors,
                            borderColor: activeColors,
                            borderWidth: 1,
                            borderRadius: 6
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        scales: {
                            y: { beginAtZero: true, ticks: { color: '#94a3b8', stepSize: 1 }, grid: { color: 'rgba(255, 255, 255, 0.05)' } },
                            x: { ticks: { color: '#e2e8f0' }, grid: { display: false } }
                        },
                        plugins: {
                            legend: { display: false }
                        }
                    }
                });
            }'''

new_chart_logic = '''            const ctx = document.getElementById('incidenciasChart').getContext('2d');
            if (incidenciasChart) {
                incidenciasChart.data.labels = activeLabels;
                incidenciasChart.data.datasets[0].data = activeValues;
                incidenciasChart.data.datasets[0].backgroundColor = activeColors;
                incidenciasChart.data.datasets[0].borderColor = activeColors;
                if (incidenciasChart.options.scales.x.title) {
                    incidenciasChart.options.scales.x.title.text = Sucursal: ;
                }
                incidenciasChart.update();
            } else {
                incidenciasChart = new Chart(ctx, {
                    type: 'bar',
                    data: {
                        labels: activeLabels,
                        datasets: [{
                            label: 'Tickets Emitidos',
                            data: activeValues,
                            backgroundColor: activeColors,
                            borderColor: activeColors,
                            borderWidth: 1,
                            borderRadius: 6,
                            maxBarThickness: 45
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        scales: {
                            y: { beginAtZero: true, ticks: { color: '#94a3b8', stepSize: 1 }, grid: { color: 'rgba(255, 255, 255, 0.05)' } },
                            x: { 
                                ticks: { color: '#e2e8f0' }, 
                                grid: { display: false },
                                title: {
                                    display: true,
                                    text: Sucursal: ,
                                    color: '#94a3b8',
                                    font: { size: 14, weight: 'bold' }
                                }
                            }
                        },
                        plugins: {
                            legend: { display: false }
                        }
                    }
                });
            }'''

code = code.replace(old_chart_logic, new_chart_logic)

with open("src/cloud_backend/templates/dashboard.html", "w", encoding="utf-8") as f:
    f.write(code)

print("Chart patched successfully")
